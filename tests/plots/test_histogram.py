"""Tests for the histogram plot function."""

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyretailscience.plots.histogram import apply_range_clipping, plot
from pyretailscience.style import graph_utils as gu


# Fixture for mocking graph_utils functions
@pytest.fixture()
def _mock_gu_functions(mocker):
    """Mock the graph_utils functions."""
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


# Fixture for mocking pandas.DataFrame.plot
@pytest.fixture()
def _mock_dataframe_plot(mocker):
    """Mock the pandas.DataFrame.plot function."""
    mocker.patch("pandas.DataFrame.plot")


@pytest.fixture()
def _mock_get_base_cmap(mocker):
    """Mock the get_base_cmap function to return a custom color generator."""
    # Create a mock generator that cycles through the colors you want to use for testing
    colors = ["#FF0000", "#00FF00", "#0000FF"]  # Custom mock colors
    color_gen = cycle(colors)  # Cycle through the color list indefinitely
    mocker.patch("pyretailscience.style.tailwind.get_base_cmap", return_value=color_gen)


# Fixture for providing a sample dataframe for testing
@pytest.fixture()
def sample_dataframe():
    """A sample dataframe for testing."""
    # Example dataframe for tests
    return pd.DataFrame(
        {
            "quantity": [1, 2, 3, 4, 5],
            "revenue": [15, 25, 30, 44, 52],
            "category": ["A", "A", "B", "B", "B"],
        },
    )


# Fixture for providing a sample pandas series for testing
@pytest.fixture()
def sample_series():
    """A sample series for testing."""
    # Example series for tests
    return pd.Series([1, 2, 3, 4, 5])


# Test cases using the fixtures
@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions", "_mock_dataframe_plot")
def test_plot_single_histogram_no_legend(sample_dataframe, mocker):
    """Test the plot function with a single histogram and no legend."""
    # Create the plot axis using plt.subplots()
    _, ax = plt.subplots()

    # Call the plot function
    resulted_ax = plot(df=sample_dataframe, value_col="quantity", ax=ax)

    # Verify that df.plot was called with correct parameters
    pd.DataFrame.plot.assert_called_once_with(kind="hist", ax=ax, legend=False, color=mocker.ANY)

    # Verify that standard_graph_styles was called with correct parameters
    gu.standard_graph_styles.assert_called_once_with(
        ax=resulted_ax,
        title=None,
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=False,
    )

    # Verify that add_source_text was not called
    gu.add_source_text.assert_not_called()


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions", "_mock_dataframe_plot")
def test_plot_multiple_histograms_with_legend_when_group_col_is_not_nan(sample_dataframe, mocker):
    """Test the plot function with multiple histograms and a legend when group_col is not None."""
    # Create the plot axis using plt.subplots()
    _, ax = plt.subplots()

    # Call the plot function with grouping
    resulted_ax = plot(df=sample_dataframe, value_col="quantity", group_col="category", ax=ax)

    # Verify that df.plot was called for multiple histograms with correct parameters
    pd.DataFrame.plot.assert_called_once_with(kind="hist", ax=ax, legend=True, color=mocker.ANY)

    # Verify that standard_graph_styles was called with correct parameters
    gu.standard_graph_styles.assert_called_once_with(
        ax=resulted_ax,
        title=None,
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=False,
    )

    # Verify that add_source_text was not called
    gu.add_source_text.assert_not_called()


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions", "_mock_dataframe_plot")
def test_plot_multiple_histograms_with_legend_when_value_col_is_a_list(sample_dataframe, mocker):
    """Test the plot function with multiple histograms and a legend when value_col is a list."""
    # Create the plot axis using plt.subplots()
    _, ax = plt.subplots()

    # Call the plot function with grouping
    resulted_ax = plot(df=sample_dataframe, value_col=["quantity", "category"], ax=ax)

    # Verify that df.plot was called for multiple histograms with correct parameters
    pd.DataFrame.plot.assert_called_once_with(kind="hist", ax=ax, legend=True, color=mocker.ANY)

    # Verify that standard_graph_styles was called with correct parameters
    gu.standard_graph_styles.assert_called_once_with(
        ax=resulted_ax,
        title=None,
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=False,
    )

    # Verify that add_source_text was not called
    gu.add_source_text.assert_not_called()


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions", "_mock_dataframe_plot")
def test_plot_with_source_text(sample_dataframe, mocker):
    """Test the plot function with source text."""
    # Create the plot axis using plt.subplots()
    _, ax = plt.subplots()

    # Call the plot function with source text
    resulted_ax = plot(df=sample_dataframe, value_col="quantity", ax=ax, source_text="Source: Test Data")

    # Verify that df.plot was called with correct parameters
    pd.DataFrame.plot.assert_called_once_with(kind="hist", ax=ax, legend=False, color=mocker.ANY)

    # Verify that standard_graph_styles was called with correct parameters
    gu.standard_graph_styles.assert_called_once_with(
        ax=resulted_ax,
        title=None,
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=False,
    )

    # Verify that add_source_text was called with the correct source_text
    gu.add_source_text.assert_called_once_with(ax=resulted_ax, source_text="Source: Test Data")


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions", "_mock_dataframe_plot")
def test_plot_custom_labels(sample_dataframe, mocker):
    """Test the plot function with custom x and y labels."""
    # Create the plot axis using plt.subplots()
    _, ax = plt.subplots()

    # Call the plot function with custom labels for x and y axes
    resulted_ax = plot(df=sample_dataframe, value_col="quantity", ax=ax, x_label="Custom X", y_label="Custom Y")

    # Verify that df.plot was called with correct parameters
    pd.DataFrame.plot.assert_called_once_with(kind="hist", ax=ax, legend=False, color=mocker.ANY)

    # Verify that standard_graph_styles was called with custom x and y labels
    gu.standard_graph_styles.assert_called_once_with(
        ax=resulted_ax,
        title=None,
        x_label="Custom X",
        y_label="Custom Y",
        legend_title=None,
        move_legend_outside=False,
    )

    # Verify that add_source_text was not called
    gu.add_source_text.assert_not_called()


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions", "_mock_dataframe_plot")
def test_plot_with_series(sample_series, mocker):
    """Test the plot function with a pandas series."""
    # Create the plot axis using plt.subplots()
    _, ax = plt.subplots()

    # Call the plot function with a series (instead of dataframe and value_col)
    resulted_ax = plot(df=sample_series, ax=ax)

    # Verify that pd.Series.plot was called with correct parameters
    pd.DataFrame.plot.assert_called_once_with(kind="hist", ax=ax, legend=False, color=mocker.ANY)

    # Verify that standard_graph_styles was called with correct parameters
    gu.standard_graph_styles.assert_called_once_with(
        ax=resulted_ax,
        title=None,
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=False,
    )

    # Verify that add_source_text was not called
    gu.add_source_text.assert_not_called()


# Test cases for the apply_range_clipping function
@pytest.mark.usefixtures("sample_dataframe")
def test_apply_range_clipping_clip_lower(sample_dataframe):
    """Test apply_range_clipping with a lower bound."""
    # Call apply_range_clipping with a lower bound
    result_df = apply_range_clipping(
        sample_dataframe,
        value_col=["quantity"],
        range_lower=3,
        range_upper=None,
        range_method="clip",
    )

    # Assert that values below 3 are clipped
    expected_df = pd.DataFrame(
        {
            "quantity": [3, 3, 3, 4, 5],
            "revenue": [15, 25, 30, 44, 52],
            "category": ["A", "A", "B", "B", "B"],
        },
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("sample_dataframe")
def test_apply_range_clipping_clip_upper(sample_dataframe):
    """Test apply_range_clipping with an upper bound."""
    # Call apply_range_clipping with an upper bound
    result_df = apply_range_clipping(
        sample_dataframe,
        value_col=["quantity"],
        range_lower=None,
        range_upper=3,
        range_method="clip",
    )

    # Assert that values above 3 are clipped
    expected_df = pd.DataFrame(
        {
            "quantity": [1, 2, 3, 3, 3],
            "revenue": [15, 25, 30, 44, 52],
            "category": ["A", "A", "B", "B", "B"],
        },
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("sample_dataframe")
def test_apply_range_clipping_fillna_lower_upper(sample_dataframe):
    """Test apply_range_clipping with both lower and upper bounds using the fillna method."""
    # Call apply_range_clipping with both lower and upper bounds and use fillna method
    result_df = apply_range_clipping(
        sample_dataframe,
        value_col=["quantity"],
        range_lower=2,
        range_upper=4,
        range_method="fillna",
    )

    # Assert that values outside the range are replaced with NaN
    expected_df = pd.DataFrame(
        {
            "quantity": [np.nan, 2, 3, 4, np.nan],
            "revenue": [15, 25, 30, 44, 52],
            "category": ["A", "A", "B", "B", "B"],
        },
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("sample_dataframe")
def test_apply_range_clipping_no_bounds(sample_dataframe):
    """Test apply_range_clipping with no bounds."""
    # Call apply_range_clipping with no bounds
    result_df = apply_range_clipping(
        sample_dataframe,
        value_col=["quantity"],
        range_lower=None,
        range_upper=None,
        range_method="clip",
    )

    # Assert that the dataframe remains unchanged
    pd.testing.assert_frame_equal(result_df, sample_dataframe)


@pytest.mark.usefixtures("sample_dataframe")
def test_apply_range_clipping_clip_lower_upper(sample_dataframe):
    """Test apply_range_clipping with both lower and upper bounds using the clip method."""
    # Call apply_range_clipping with both lower and upper bounds using the clip method
    result_df = apply_range_clipping(
        sample_dataframe,
        value_col=["quantity"],
        range_lower=2,
        range_upper=4,
        range_method="clip",
    )

    # Assert that values outside the bounds are clipped
    expected_df = pd.DataFrame(
        {
            "quantity": [2, 2, 3, 4, 4],
            "revenue": [15, 25, 30, 44, 52],
            "category": ["A", "A", "B", "B", "B"],
        },
    )
    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("_mock_get_base_cmap", "sample_dataframe")
def test_plot_missing_value_col_raises_error(sample_dataframe):
    """Test the plot function raises an error when value_col is missing."""
    # Expect the plot function to raise a ValueError if value_col is missing
    with pytest.raises(ValueError, match="Please provide a value column to plot."):
        # Call the plot function without value_col
        plot(df=sample_dataframe, value_col=None)


@pytest.mark.usefixtures("_mock_get_base_cmap", "sample_dataframe")
def test_plot_value_col_list_and_group_col_raises_error():
    """Test the plot function raises an error when value_col is a list and group_col is provided."""
    # ValueError should be raised when `value_col` is a list and `group_col` is provided
    with pytest.raises(ValueError, match="`value_col` cannot be a list when `group_col` is provided."):
        plot(df=sample_dataframe, value_col=["quantity", "quantity"], group_col="category")
