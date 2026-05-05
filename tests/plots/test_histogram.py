"""Tests for the histograms plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.options import PlotStyleHelper
from openretailscience.plots import histogram


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "value_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "group": ["A"] * 5 + ["B"] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_series():
    """A sample series for testing."""
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_plot_single_histogram(sample_dataframe):
    """Test the plot function with a single histogram."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Single Histogram",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


def test_plot_grouped_histogram(sample_dataframe):
    """Test the plot function with grouped histograms."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col="group",
        title="Test Grouped Histogram",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


def test_plot_enforces_range_clipping(sample_dataframe):
    """Test that the plot function enforces range clipping through the Axes limits and print the min/max values."""
    range_lower = 2
    range_upper = 8

    # Plot with range clipping
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Histogram with Range Clipping",
        range_lower=range_lower,
        range_upper=range_upper,
        range_method="clip",
    )

    # Get the data limits from the resulting Axes object
    x_data = result_ax.patches  # Access the bars in the histogram
    clipped_values = [patch.get_x() for patch in x_data]

    # Ensure that the x values (bars' positions) respect the clipping limits
    assert all(range_lower <= val + np.finfo(np.float64).eps <= range_upper for val in clipped_values)


def test_plot_with_range_fillna(sample_dataframe):
    """Test the plot function with range fillna."""
    range_lower = 3
    range_upper = 9

    # Plot with range clipping
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Histogram with Range Clipping",
        range_lower=range_lower,
        range_upper=range_upper,
        range_method="fillna",
    )

    # Get the data limits from the resulting Axes object
    x_data = result_ax.patches  # Access the bars in the histogram
    clipped_values = [patch.get_x() for patch in x_data]

    # Ensure that the x values (bars' positions) respect the clipping limits
    assert all(range_lower <= val + np.finfo(np.float64).eps <= range_upper for val in clipped_values)


def test_plot_with_range_lower_none(sample_dataframe):
    """Test the plot function with range_lower=None (no lower bound) and a specific upper bound."""
    range_upper = 8  # No lower bound

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Histogram with Upper Bound Only",
        range_lower=None,
        range_upper=range_upper,
        range_method="clip",
    )

    # Get the data limits from the resulting Axes object
    x_data = result_ax.patches
    clipped_values = [patch.get_x() for patch in x_data]

    # Ensure that the x values (bars' positions) respect the upper bound, but no lower bound is applied
    assert all(val + np.finfo(np.float64).eps <= range_upper for val in clipped_values)


def test_plot_with_range_upper_none(sample_dataframe):
    """Test the plot function with range_upper=None (no upper bound) and a specific lower bound."""
    range_lower = 3  # No upper bound

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Histogram with Lower Bound Only",
        range_lower=range_lower,
        range_upper=None,
        range_method="clip",
    )

    # Get the data limits from the resulting Axes object
    x_data = result_ax.patches
    clipped_values = [patch.get_x() for patch in x_data]

    # Ensure that the x values (bars' positions) respect the lower bound, but no upper bound is applied
    assert all(range_lower <= val + np.finfo(np.float64).eps for val in clipped_values)


def test_plot_fillna_outside_range(sample_dataframe):
    """Test the fillna method, ensuring values outside the range are replaced by NaN."""
    range_lower = 3
    range_upper = 8

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Histogram with Range Fillna",
        range_lower=range_lower,
        range_upper=range_upper,
        range_method="fillna",
    )

    # Extract data from the resulting Axes
    x_data = result_ax.patches
    clipped_values = [patch.get_x() for patch in x_data]

    # Ensure that values outside the range are not plotted (NaN)
    assert all(range_lower <= val + np.finfo(np.float64).eps <= range_upper for val in clipped_values)


def test_plot_single_histogram_series(sample_series):
    """Test the plot function with a pandas series."""
    result_ax = histogram.plot(
        df=sample_series,
        title="Test Single Histogram (Series)",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


def test_plot_histogram_with_hatch(sample_dataframe):
    """use_hatch=True applies a hatch pattern to every histogram patch."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test Histogram with Hatch",
        use_hatch=True,
    )

    hatches = [p.get_hatch() for p in result_ax.patches]
    assert len(hatches) > 0
    assert all(h is not None for h in hatches)


def test_plot_invalid_value_col_with_group_col(sample_dataframe):
    """Test the plot function raises an error when both `value_col` is a list and `group_col` is provided."""
    with pytest.raises(ValueError, match="`value_col` cannot be a list when `group_col` is provided"):
        histogram.plot(
            df=sample_dataframe,
            value_col=["value_1", "value_2"],
            group_col="group",
            title="Test Invalid Value Col with Group Col",
        )


def test_plot_legend_outside(sample_dataframe):
    """move_legend_outside=True anchors the legend at the configured outside position."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col="group",
        title="Test Legend Outside",
        move_legend_outside=True,
    )

    legend = result_ax.get_legend()
    assert legend is not None
    anchor = legend.get_bbox_to_anchor().transformed(result_ax.transAxes.inverted())
    expected_x, expected_y = PlotStyleHelper().legend_bbox_to_anchor
    assert anchor.x0 == pytest.approx(expected_x)
    assert anchor.y0 == pytest.approx(expected_y)


def test_plot_adds_source_text(sample_dataframe):
    """The histogram renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        title="Test with Source Text",
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_plot_multiple_histograms(sample_dataframe):
    """Test the plot function with multiple histograms."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col=["value_1", "value_2"],
        title="Test Multiple Histograms",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that bars were plotted for both histograms


@pytest.mark.parametrize(
    ("group_col", "expected_legend", "expected_alpha"),
    [
        (None, False, None),
        ("group", True, 0.7),
    ],
)
def test_histogram_legend_and_alpha_by_grouping(sample_dataframe, group_col, expected_legend, expected_alpha):
    """Single histograms render without a legend/alpha; grouped histograms render with both."""
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col=group_col,
        title="Histogram grouping",
    )

    assert (result_ax.get_legend() is not None) is expected_legend

    patch_alphas = {p.get_alpha() for p in result_ax.patches}
    assert patch_alphas == {expected_alpha}
