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
    """Test heatmap with minimal DataFrame dimensions: single row, single column, and single cell."""
    rows, cols = shape
    data = np.round(RNG.uniform(0, 1, size=shape), 2)
    df = pd.DataFrame(data, columns=[f"Col {i}" for i in range(cols)], index=[f"Row {i}" for i in range(rows)])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_xticks()) == cols
    assert len(result_ax.get_yticks()) == rows
    assert len(result_ax.texts) == df.size


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
        threshold = 10
        expected_rotation = 45
        if len(label_length) > threshold:
            assert rotation == expected_rotation, "Long labels should be rotated"
        else:
            assert rotation == 0, "Short labels should not be rotated"


def test_plot_label_alignment():
    """Test horizontal alignment of x-axis labels based on rotation."""
    short_cols = ["A", "B", "C"]
    data = np.ones((2, 3))
    df_short = pd.DataFrame(data, columns=short_cols, index=["Row1", "Row2"])

    result_ax = heatmap.plot(df=df_short, cbar_label="Value")
    x_tick_labels = result_ax.get_xticklabels()

    if x_tick_labels:
        alignment = x_tick_labels[0].get_horizontalalignment()
        assert alignment == "center", "Short labels should be center-aligned"

    # Test with long labels (rotated)
    long_cols = ["very_long_column_name_1", "very_long_column_name_2", "very_long_column_name_3"]
    df_long = pd.DataFrame(data, columns=long_cols, index=["Row1", "Row2"])

    result_ax = heatmap.plot(df=df_long, cbar_label="Value")
    x_tick_labels = result_ax.get_xticklabels()

    if x_tick_labels:
        alignment = x_tick_labels[0].get_horizontalalignment()
        assert alignment == "right", "Long rotated labels should be right-aligned"


def test_colorbar_label_set(sample_heatmap_dataframe):
    """Verify colorbar label is set correctly."""
    label = "Test Colorbar Label"
    result_ax = heatmap.plot(df=sample_heatmap_dataframe, cbar_label=label)

    # Get colorbar axes (should be the last axes in figure)
    cbar_ax = result_ax.figure.axes[-1]
    # Check ylabel
    ylabel = cbar_ax.get_ylabel()
    assert ylabel == label, f"Expected colorbar label '{label}', got '{ylabel}'"


def test_axis_labels_applied(sample_heatmap_dataframe):
    """Verify axis labels and title are applied correctly."""
    result_ax = heatmap.plot(
        df=sample_heatmap_dataframe,
        cbar_label="Value",
        x_label="X Axis Label",
        y_label="Y Axis Label",
        title="Test Title",
    )

    assert result_ax.get_xlabel() == "X Axis Label"
    assert result_ax.get_ylabel() == "Y Axis Label"
    assert result_ax.get_title() == "Test Title"


def test_text_color_contrast():
    """Verify text color switches based on cell background intensity."""
    # Create data with known light and dark cells
    data = np.array([[0.0, 1.0]])  # Dark cell, light cell
    df = pd.DataFrame(data, columns=["A", "B"], index=["X"])

    result_ax = heatmap.plot(df=df, cbar_label="Value")

    texts = result_ax.texts
    expected_text_count = 2
    assert len(texts) == expected_text_count

    # Get text colors
    text_0_color = texts[0].get_color()
    text_1_color = texts[1].get_color()

    # Verify colors are different (contrast based on background intensity)
    assert text_0_color != text_1_color, "Text colors should differ for different background intensities"
