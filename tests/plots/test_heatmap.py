"""Tests for the heatmap plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import heatmap
from pyretailscience.plots.styles import graph_utils as gu

RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_heatmap_dataframe():
    """Generates a sample DataFrame for heatmap testing."""
    data = np.round(RNG.uniform(0, 1, size=(4, 4)), 2)
    return pd.DataFrame(
        data,
        columns=[f"Col {i + 1}" for i in range(4)],
        index=[f"Row {i + 1}" for i in range(4)],
    )


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mocks graph utility functions to avoid modifying global styles."""
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


def test_plot_basic_heatmap(sample_heatmap_dataframe):
    """Test basic heatmap creation with required parameters."""
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        title="Basic Heatmap Test",
    )

    assert isinstance(result_ax, Axes)
    # Verify colorbar exists (plot + colorbar = 2 axes)
    expected_axis_count = 2
    assert len(result_ax.figure.axes) == expected_axis_count
    # Verify text elements for each cell
    assert len(result_ax.texts) == sample_heatmap_dataframe.size


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_with_source_text(sample_heatmap_dataframe):
    """Test heatmap with source text annotation."""
    source_text = "Source: Test Data"
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


def test_plot_with_figsize(sample_heatmap_dataframe):
    """Test heatmap with custom figure size."""
    width, height = 12, 8
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        figsize=(width, height),
    )

    assert result_ax.figure.get_size_inches()[0] == width
    assert result_ax.figure.get_size_inches()[1] == height


def test_plot_with_kwargs(sample_heatmap_dataframe):
    """Test heatmap with additional kwargs."""
    test_alpha = 0.7
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Test Values",
        alpha=test_alpha,
        interpolation="nearest",
    )

    # Verify kwargs are passed to imshow
    images = result_ax.get_images()
    assert len(images) > 0
    assert images[0].get_alpha() == test_alpha


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
def test_plot_edge_case_dimensions(shape):
    """Test heatmap with edge case DataFrame dimensions."""
    rows, cols = shape
    data = np.round(RNG.uniform(0, 1, size=shape), 2)
    df = pd.DataFrame(data, columns=[f"Col {i}" for i in range(cols)], index=[f"Row {i}" for i in range(rows)])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_xticks()) == cols
    assert len(result_ax.get_yticks()) == rows
    assert len(result_ax.texts) == df.size


def test_plot_empty_dataframe():
    """Test heatmap with empty DataFrame raises error."""
    empty_df = pd.DataFrame()

    with pytest.raises((ValueError, IndexError)):
        heatmap.plot(df=empty_df, cbar_label="Value")


def test_plot_text_values_accuracy(sample_heatmap_dataframe):
    """Test that displayed text values match DataFrame values."""
    result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label="Test Values")

    texts = result_ax.texts
    for i, text in enumerate(texts):
        row, col = divmod(i, sample_heatmap_dataframe.shape[1])
        expected_value = sample_heatmap_dataframe.iloc[row, col]
        displayed_value = float(text.get_text())
        value_tolerance = 0.01
        assert abs(displayed_value - expected_value) < value_tolerance


@pytest.mark.parametrize("data_range", [(0, 1), (-1, 1), (100, 200)])
def test_plot_different_value_ranges(data_range):
    """Test heatmap with different data value ranges."""
    min_val, max_val = data_range
    data = np.round(RNG.uniform(min_val, max_val, size=(3, 3)), 2)
    df = pd.DataFrame(data, columns=[f"Col {i}" for i in range(3)], index=[f"Row {i}" for i in range(3)])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    # Verify all displayed values are within expected range
    texts = result_ax.texts
    text_values = [float(text.get_text()) for text in texts]
    assert all(min_val <= val <= max_val for val in text_values)


def test_plot_with_nan_values():
    """Test heatmap handles NaN values gracefully."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0]])
    df = pd.DataFrame(data, columns=["A", "B"], index=["X", "Y"])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    # Verify NaN is displayed in text
    texts = result_ax.texts
    nan_found = any("nan" in text.get_text().lower() for text in texts)
    assert nan_found, "NaN value should be displayed in heatmap text"


def test_plot_all_zeros():
    """Test heatmap with all-zero data."""
    data = np.zeros((2, 2))
    df = pd.DataFrame(data, columns=["A", "B"], index=["X", "Y"])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    # Verify all text shows "0.00"
    texts = result_ax.texts
    assert all(text.get_text() == "0.00" for text in texts)


@pytest.mark.parametrize("label_length", ["short", "very_long_column_name_that_exceeds_threshold"])
def test_plot_label_rotation(label_length):
    """Test automatic label rotation based on label length."""
    cols = [label_length] * 3
    data = np.ones((2, 3))
    df = pd.DataFrame(data, columns=cols, index=["A", "B"])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    x_tick_labels = result_ax.get_xticklabels()
    if x_tick_labels:
        rotation = x_tick_labels[0].get_rotation()
        long_label_threshold = 10
        rotation_angle = 45
        if len(label_length) > long_label_threshold:
            assert rotation == rotation_angle, "Long labels should be rotated"
        else:
            assert rotation == 0, "Short labels should not be rotated"
