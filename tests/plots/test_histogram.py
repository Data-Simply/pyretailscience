"""Tests for the histograms plot module."""

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import histogram
from pyretailscience.style import graph_utils as gu


@pytest.fixture()
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "value_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "value_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "group": ["A"] * 5 + ["B"] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_series():
    """A sample series for testing."""
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture()
def _mock_color_generators(mocker):
    """Mock the color generator for multi color maps."""
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])  # Mocked multi-color generator
    mocker.patch("pyretailscience.style.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture()
def _mock_gu_functions(mocker):
    """Mock standard graph utilities functions."""
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)
    mocker.patch("pyretailscience.style.graph_utils.apply_hatches", side_effect=lambda ax, num_segments: ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_histogram(sample_dataframe):
    """Test the plot function with a single histogram."""
    _, ax = plt.subplots()

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        ax=ax,
        title="Test Single Histogram",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_grouped_histogram(sample_dataframe):
    """Test the plot function with grouped histograms."""
    _, ax = plt.subplots()

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col="group",
        ax=ax,
        title="Test Grouped Histogram",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_enforces_range_clipping(sample_dataframe):
    """Test that the plot function enforces range clipping through the Axes limits and print the min/max values."""
    _, ax = plt.subplots()
    range_lower = 2
    range_upper = 8

    # Plot with range clipping
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        ax=ax,
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_range_fillna(sample_dataframe, mocker):
    """Test the plot function with range fillna."""
    _, ax = plt.subplots()
    range_lower = 3
    range_upper = 9

    # Plot with range clipping
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        ax=ax,
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_histogram_series(sample_series):
    """Test the plot function with a pandas series."""
    _, ax = plt.subplots()

    result_ax = histogram.plot(
        df=sample_series,
        ax=ax,
        title="Test Single Histogram (Series)",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_histogram_with_hatch(sample_dataframe):
    """Test the plot function with hatching."""
    _, ax = plt.subplots()

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        ax=ax,
        title="Test Histogram with Hatch",
        use_hatch=True,
    )

    gu.apply_hatches.assert_called_once_with(ax=result_ax, num_segments=1)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_invalid_value_col_with_group_col(sample_dataframe):
    """Test the plot function raises an error when both `value_col` is a list and `group_col` is provided."""
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="`value_col` cannot be a list when `group_col` is provided"):
        histogram.plot(
            df=sample_dataframe,
            value_col=["value_1", "value_2"],
            group_col="group",
            ax=ax,
            title="Test Invalid Value Col with Group Col",
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_legend_outside(sample_dataframe):
    """Test the plot function moves the legend outside the plot."""
    _, ax = plt.subplots()

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        group_col="group",
        ax=ax,
        title="Test Legend Outside",
        move_legend_outside=True,
    )

    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title="Test Legend Outside",
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=True,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(sample_dataframe):
    """Test the plot function adds source text to the plot."""
    _, ax = plt.subplots()

    source_text = "Source: Test Data"
    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col="value_1",
        ax=ax,
        title="Test with Source Text",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_histograms(sample_dataframe):
    """Test the plot function with multiple histograms."""
    _, ax = plt.subplots()

    result_ax = histogram.plot(
        df=sample_dataframe,
        value_col=["value_1", "value_2"],
        ax=ax,
        title="Test Multiple Histograms",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that bars were plotted for both histograms


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_histogram_calls_dataframe_plot(mocker, sample_dataframe):
    """Test that pandas.DataFrame.plot is called with correct arguments when group_col is None."""
    # Spy on DataFrame.plot
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    # Prepare input
    _, ax = plt.subplots()
    value_col = "value_1"
    range_lower = 2
    range_upper = 8

    # Call the histogram plot function without a group column (single histogram)
    histogram.plot(
        df=sample_dataframe,
        value_col=value_col,
        ax=ax,
        range_lower=range_lower,
        range_upper=range_upper,
        title="Test Single Histogram",
    )

    # Check the arguments passed to DataFrame.plot for single histogram
    mock_df_plot.assert_called_once_with(
        kind="hist",
        ax=mocker.ANY,
        legend=False,  # No grouping, so no need for a legend
        color=mocker.ANY,
        alpha=None,  # Since it's a single histogram
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_grouped_histogram_calls_dataframe_plot(mocker, sample_dataframe):
    """Test that pandas.DataFrame.plot is called with correct arguments when group_col is provided."""
    # Spy on DataFrame.plot
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    # Prepare input
    _, ax = plt.subplots()
    value_col = "value_1"
    group_col = "group"
    range_lower = 2
    range_upper = 8

    # Call the histogram plot function with a group column (multiple histograms)
    histogram.plot(
        df=sample_dataframe,
        value_col=value_col,
        group_col=group_col,
        ax=ax,
        range_lower=range_lower,
        range_upper=range_upper,
        title="Test Grouped Histogram",
    )

    # Check the arguments passed to DataFrame.plot for grouped histogram
    mock_df_plot.assert_called_once_with(
        kind="hist",
        ax=mocker.ANY,
        legend=True,  # Multiple histograms should have a legend
        color=mocker.ANY,
        alpha=0.7,  # Alpha is set to 0.7 for grouped histograms
    )
