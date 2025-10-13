"""Tests for the cohort plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import cohort
from pyretailscience.plots.styles import graph_utils as gu

RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_cohort_dataframe():
    """Generates a sample cohort DataFrame."""
    data = np.round(RNG.uniform(0, 1, size=(6, 6)), 2)
    return pd.DataFrame(data, columns=[f"Month {i + 1}" for i in range(6)], index=[f"Cohort {i + 1}" for i in range(6)])


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mocks graph utility functions to avoid modifying global styles."""
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch(
        "pyretailscience.plots.styles.graph_utils.add_source_text",
        side_effect=lambda ax, source_text: ax,
    )


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort(sample_cohort_dataframe):
    """Test cohort plot with a standard DataFrame."""
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        x_label="Months",
        y_label="Cohorts",
        title="Cohort Retention Heatmap",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_source_text(sample_cohort_dataframe):
    """Test cohort plot with source text annotation."""
    source_text = "Source: Test Data"

    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_percentage(sample_cohort_dataframe):
    """Test cohort plot with percentage formatting enabled."""
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        percentage=True,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_ax_none(sample_cohort_dataframe):
    """Test cohort plot when ax is None (should create a new figure)."""
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        ax=None,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0
    assert result_ax.figure is not None


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_figsize(sample_cohort_dataframe):
    """Test cohort plot with a specified figsize."""
    width = 14
    height = 10
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        figsize=(width, height),
    )

    assert isinstance(result_ax, Axes)
    assert result_ax.figure.get_size_inches()[0] == width
    assert result_ax.figure.get_size_inches()[1] == height


@pytest.mark.parametrize("percentage", [True, False])
def test_plot_cohort_percentage_formatting(percentage):
    """Test cohort plot percentage formatting comprehensively."""
    data = np.array([[0.5, 0.3], [0.8, 0.6]])
    df = pd.DataFrame(data, columns=["Month 1", "Month 2"], index=["Cohort A", "Cohort B"])

    result_ax = cohort.plot(df=df, cbar_label="Retention Rate", percentage=percentage)

    assert len(result_ax.texts) > 0, "Should have cell text elements"
    texts = result_ax.texts

    if percentage:
        assert any("%" in text.get_text() for text in texts), "Should have percentage formatting"
        # Check specific formatting (50% not 0.50)
        text_values = [text.get_text() for text in texts]
        # At least one should be a proper percentage format like "50%" not "0.50"
        proper_percentage_found = any("%" in val and not val.startswith("0.") for val in text_values)
        assert proper_percentage_found, f"Expected proper percentage format, got: {text_values}"
    else:
        assert all("%" not in text.get_text() for text in texts), "Should not have percentage formatting"


def test_cohort_horizontal_line_exists(sample_cohort_dataframe):
    """Verify cohort-specific horizontal line is drawn."""
    result_ax = cohort.plot(df=sample_cohort_dataframe, cbar_label="Retention Rate")

    # Check that horizontal lines exist
    # hlines() creates LineCollections, not Line2D objects
    collections = result_ax.collections
    line_collections = [col for col in collections if hasattr(col, "get_segments")]
    assert len(line_collections) > 0, "Cohort plot should have horizontal line collection"


def test_cohort_horizontal_line_position():
    """Verify horizontal line is at correct y-position."""
    df = pd.DataFrame(np.ones((6, 4)))
    result_ax = cohort.plot(df=df, cbar_label="Rate")

    # Get line collections
    line_collections = [col for col in result_ax.collections if hasattr(col, "get_segments")]
    assert len(line_collections) > 0

    # Verify line position (should be at y=2.5 for hardcoded value 3)
    segments = line_collections[0].get_segments()
    assert len(segments) > 0, "Line collection should have segments"

    # Check that line y-coordinate is 2.5 (hardcoded as y=3-0.5 in cohort.py:85)
    y_position = segments[0][0][1]  # First segment, first point, y-coordinate
    expected_y_position = 2.5
    assert y_position == expected_y_position, f"Expected line at y={expected_y_position}, got y={y_position}"
