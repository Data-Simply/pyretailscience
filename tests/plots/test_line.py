"""Tests for the plots.line module."""

import warnings
from itertools import cycle

import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import line
from pyretailscience.style import graph_utils as gu


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "x": pd.date_range("2023-01-01", periods=10, freq="D"),
        "y": range(10, 20),
        "group": ["A"] * 5 + ["B"] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi color maps."""
    single_color_gen = cycle(["#FF0000"])  # Mocked single-color generator (e.g., red)
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])  # Mocked multi-color generator (red, green, blue)

    mocker.patch("pyretailscience.style.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.style.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture
def _mock_gu_functions(mocker):
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_group_col(sample_dataframe):
    """Test the plot function with a group column."""
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot",
        x_col="x",
        group_col="group",
    )
    expected_num_lines = 2

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_lines()) == expected_num_lines  # One line for each group


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_without_group_col(sample_dataframe):
    """Test the plot function without a group column."""
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Without Group",
        x_col="x",
    )
    expected_num_lines = 1

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_lines()) == expected_num_lines  # Only one line


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_warns_if_xcol_is_datetime(sample_dataframe, mocker):
    """Test the plot function warns if the x_col is datetime-like."""
    mocker.patch("warnings.warn")

    line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Datetime Warning",
        x_col="x",
        group_col="group",
    )

    warnings.warn.assert_any_call(
        "The column 'x' is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
        UserWarning,
        stacklevel=2,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_moves_legend_outside(sample_dataframe):
    """Test the plot function moves the legend outside the plot."""
    # Create the plot with move_legend_outside=True
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Legend Outside",
        x_col="x",
        group_col="group",
        move_legend_outside=True,
    )

    # Assert that standard_graph_styles was called with move_legend_outside=True
    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title="Test Plot Legend Outside",
        x_label="X Axis",
        y_label="Y Axis",
        legend_title=None,
        move_legend_outside=True,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_moves_legend_inside(sample_dataframe):
    """Test the plot function moves the legend inside the plot."""
    # Create the plot with move_legend_outside=False
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Legend Inside",
        x_col="x",
        group_col="group",
        move_legend_outside=False,
    )

    # Assert that standard_graph_styles was called with move_legend_outside=False
    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title="Test Plot Legend Inside",
        x_label="X Axis",
        y_label="Y Axis",
        legend_title=None,
        move_legend_outside=False,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(sample_dataframe):
    """Test the plot function adds source text to the plot."""
    source_text = "Source: Test Data"

    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Source Text",
        x_col="x",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_legend_title(sample_dataframe):
    """Test the plot function with a legend title."""
    # Create the plot with a legend title
    legend_title = "Test Legend"
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot with Legend Title",
        x_col="x",
        group_col="group",
        legend_title=legend_title,
    )

    # Assert that standard_graph_styles was called with the provided legend title
    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title="Test Plot with Legend Title",
        x_label="X Axis",
        y_label="Y Axis",
        legend_title=legend_title,
        move_legend_outside=False,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_legend_title_and_move_outside(sample_dataframe):
    """Test the plot function with both move_legend_outside=True and legend_title."""
    # Create the plot with both options
    legend_title = "Test Legend"
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Legend Outside with Title",
        x_col="x",
        group_col="group",
        move_legend_outside=True,
        legend_title=legend_title,
    )

    # Assert that standard_graph_styles was called with both options
    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title="Test Plot Legend Outside with Title",
        x_label="X Axis",
        y_label="Y Axis",
        legend_title=legend_title,
        move_legend_outside=True,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_datetime_index_warns(sample_dataframe, mocker):
    """Test the plot function with a datetime index and no x_col, expecting a warning."""
    df_with_datetime_index = sample_dataframe.set_index("x")

    # Mock the warnings.warn method to check if it's called
    mocker.patch("warnings.warn")

    # Create the plot with a datetime index and no x_col
    result_ax = line.plot(
        df=df_with_datetime_index,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Datetime Index",
    )

    # Assert that the plot was created
    assert isinstance(result_ax, Axes)

    # Assert that the warning about datetime-like index was raised
    warnings.warn.assert_called_once_with(
        "The DataFrame index is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
        UserWarning,
        stacklevel=2,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_line_plot_single_value_col_calls_dataframe_plot(mocker, sample_dataframe):
    """Test that pandas.DataFrame.plot is called with correct arguments when group_col is None."""
    # Mock DataFrame's plot method
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    # Call the line plot function without a group column (single line)
    line.plot(
        df=sample_dataframe,
        value_col="y",
        x_col="x",
        title="Test Single Line Plot",
    )

    # Check that DataFrame.plot was called with the correct arguments
    mock_df_plot.assert_called_once_with(
        ax=mocker.ANY,
        linewidth=3,
        color=mocker.ANY,  # Dynamic color generation
        legend=False,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_line_plot_grouped_series_calls_dataframe_plot(mocker, sample_dataframe):
    """Test that pandas.DataFrame.plot is called with correct arguments when group_col is provided."""
    # Mock DataFrame's plot method
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    # Call the line plot function with a group column (multiple lines)
    line.plot(
        df=sample_dataframe,
        value_col="y",
        group_col="group",
        x_col="x",
        title="Test Grouped Line Plot",
    )

    # Check that DataFrame.plot was called with the correct arguments for grouped plot
    mock_df_plot.assert_called_once_with(
        ax=mocker.ANY,
        linewidth=3,
        color=mocker.ANY,  # Dynamic color generation
        legend=True,  # Legend should be present for grouped lines
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_columns_with_group_col(sample_dataframe):
    """Test the plot function when using multiple columns along with a group column."""
    sample_dataframe["y1"] = range(10, 20)
    with pytest.raises(ValueError, match="Cannot use both a list for `value_col` and a `group_col`. Choose one."):
        line.plot(
            df=sample_dataframe,
            value_col=["y", "y1"],
            x_label="Transaction Date",
            y_label="Sales",
            title="Sales Trend (Grouped by Category)",
            x_col="x",
            group_col="group",
        )
