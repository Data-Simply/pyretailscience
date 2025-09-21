"""Tests for the plots.line module."""

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import line
from pyretailscience.plots.styles import graph_utils as gu


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


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
def retail_sales_dataframe():
    """A realistic retail sales dataframe that creates predictable NaN values when pivoted."""
    data = {
        "week": [1, 1, 2, 2, 3, 3, 4],
        "sales": [1250.50, 890.25, 1450.75, 920.00, 1100.00, 980.30, 750.00],
        "store": [
            "Store_North",
            "Store_South",
            "Store_North",
            "Store_South",
            "Store_North",
            "Store_South",
            "Store_South",
        ],
    }
    # Store_South has complete data for weeks 1, 2, 3, 4 (no missing values)
    # Store_North has data for weeks 1, 2, 3 but missing week 4 (creates NaN when pivoted)
    return pd.DataFrame(data)


@pytest.fixture
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi color maps."""
    single_color_gen = cycle(["#FF0000"])  # Mocked single-color generator (e.g., red)
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])  # Mocked multi-color generator (red, green, blue)

    mocker.patch("pyretailscience.plots.styles.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.plots.styles.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture
def _mock_gu_functions(mocker):
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_fill_na_value_fills_missing_pivot_values(retail_sales_dataframe):
    """Test that fill_na_value fills NaN values for Store_North but not Store_South."""
    result_ax = line.plot(
        df=retail_sales_dataframe,
        value_col="sales",
        x_col="week",
        group_col="store",
        fill_na_value=0.0,
    )

    assert isinstance(result_ax, Axes)
    expected_num_lines = 2  # One for each store
    assert len(result_ax.get_lines()) == expected_num_lines

    # Get the plotted data for each store
    line_data = {}
    for plot_line in result_ax.get_lines():
        label = plot_line.get_label()
        y_data = plot_line.get_ydata()
        line_data[label] = y_data

    north_data = line_data["Store_North"]
    south_data = line_data["Store_South"]

    # Store_North should have 0.0 (filled value) for missing week 4
    assert 0.0 in north_data, "Store_North should have 0.0 for missing week 4"

    # Store_South should NOT have any 0.0 values (has complete data)
    assert 0.0 not in south_data, "Store_South should not have any 0.0 values"


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_fill_na_value_none_preserves_nan_values(retail_sales_dataframe):
    """Test that when fill_na_value is None, NaN exists in Store_North but not Store_South."""
    result_ax = line.plot(
        df=retail_sales_dataframe,
        value_col="sales",
        x_col="week",
        group_col="store",
        fill_na_value=None,
    )

    assert isinstance(result_ax, Axes)
    expected_num_lines = 2  # One for each store
    assert len(result_ax.get_lines()) == expected_num_lines

    # Get the plotted data for each store
    line_data = {}
    for plot_line in result_ax.get_lines():
        label = plot_line.get_label()
        y_data = plot_line.get_ydata()
        line_data[label] = y_data

    north_data = line_data["Store_North"]
    south_data = line_data["Store_South"]

    # When fill_na_value=None, matplotlib handles missing data with masked arrays
    # Store_North with missing data should be a masked array
    assert isinstance(north_data, np.ma.MaskedArray), "Store_North data should be a masked array due to missing data"
    assert north_data.mask.any(), "Store_North should have masked values for missing week 4"

    # Store_South with complete data should be a regular numpy array (no masking needed)
    assert isinstance(south_data, np.ndarray), "Store_South data should be a numpy array"
    assert not isinstance(south_data, np.ma.MaskedArray), "Store_South should not be a masked array (has complete data)"
