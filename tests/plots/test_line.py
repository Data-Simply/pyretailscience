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
        "The column 'x' is datetime-like. Consider using the 'timeline' module for time-based plots.",
        UserWarning,
        stacklevel=2,
    )


@pytest.mark.usefixtures("_mock_get_base_cmap", "_mock_gu_functions")
def test_plot_moves_legend_outside(sample_dataframe):
    """Test the plot function moves the legend outside the plot."""
    _, ax = plt.subplots()

    # Test with move_legend_outside=True
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

    expected_coords = (1.05, 1.0)
    legend = result_ax.get_legend()
    # Check if bbox_to_anchor is set to (1.05, 1) when legend is outside
    bbox_anchor = legend.get_bbox_to_anchor()._bbox

    assert legend is not None
    assert (bbox_anchor.x0, bbox_anchor.y0) == expected_coords


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
