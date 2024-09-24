"""Tests for the plots.line module."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from pyretailscience.plots import line
from pyretailscience.style import graph_utils as gu


@pytest.fixture()
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "x": pd.date_range("2023-01-01", periods=10, freq="D"),
        "y": range(10, 20),
        "group": ["A"] * 5 + ["B"] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture()
def _mock_get_base_cmap(mocker):
    """Mock the get_base_cmap function to return a custom colormap."""
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    mocker.patch("pyretailscience.style.tailwind.get_base_cmap", return_value=cmap)


@pytest.fixture()
def _mock_gu_functions(mocker):
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_with_group_col(sample_dataframe):
    """Test the plot function with a group column."""
    _, ax = plt.subplots()

    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot",
        x_col="x",
        group_col="group",
        ax=ax,
    )
    expected_num_lines = 2

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_lines()) == expected_num_lines  # One line for each group


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_without_group_col(sample_dataframe):
    """Test the plot function without a group column."""
    _, ax = plt.subplots()

    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Without Group",
        x_col="x",
        ax=ax,
    )
    expected_num_lines = 1

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_lines()) == expected_num_lines  # Only one line


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_warns_if_xcol_is_datetime(sample_dataframe, mocker):
    """Test the plot function warns if the x_col is datetime-like."""
    mocker.patch("warnings.warn")
    _, ax = plt.subplots()

    line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Datetime Warning",
        x_col="x",
        group_col="group",
        ax=ax,
    )

    warnings.warn.assert_called_once_with(
        "The column 'x' is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
        UserWarning,
        stacklevel=2,
    )


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_moves_legend_outside(sample_dataframe):
    """Test the plot function moves the legend outside the plot."""
    _, ax = plt.subplots()

    # Create the plot with move_legend_outside=True
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Legend Outside",
        x_col="x",
        group_col="group",
        ax=ax,
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


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_moves_legend_inside(sample_dataframe):
    """Test the plot function moves the legend inside the plot."""
    _, ax = plt.subplots()

    # Create the plot with move_legend_outside=False
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Legend Inside",
        x_col="x",
        group_col="group",
        ax=ax,
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


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_adds_source_text(sample_dataframe):
    """Test the plot function adds source text to the plot."""
    _, ax = plt.subplots()
    source_text = "Source: Test Data"

    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Source Text",
        x_col="x",
        ax=ax,
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_with_legend_title(sample_dataframe):
    """Test the plot function with a legend title."""
    _, ax = plt.subplots()

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
        ax=ax,
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


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_with_legend_title_and_move_outside(sample_dataframe):
    """Test the plot function with both move_legend_outside=True and legend_title."""
    _, ax = plt.subplots()

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
        ax=ax,
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


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_with_datetime_index_warns(sample_dataframe, mocker):
    """Test the plot function with a datetime index and no x_col, expecting a warning."""
    df_with_datetime_index = sample_dataframe.set_index("x")
    _, ax = plt.subplots()

    # Mock the warnings.warn method to check if it's called
    mocker.patch("warnings.warn")

    # Create the plot with a datetime index and no x_col
    result_ax = line.plot(
        df=df_with_datetime_index,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Datetime Index",
        ax=ax,
    )

    # Assert that the plot was created
    assert isinstance(result_ax, Axes)

    # Assert that the warning about datetime-like index was raised
    warnings.warn.assert_called_once_with(
        "The DataFrame index is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
        UserWarning,
        stacklevel=2,
    )
