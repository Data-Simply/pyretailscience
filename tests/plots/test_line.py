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
def sample_series():
    """A sample pandas Series for testing."""
    return pd.Series([100, 150, 200, 175, 225], index=["A", "B", "C", "D", "E"], name="revenue")


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

    assert isinstance(result_ax, Axes), "Result should be an Axes object"
    assert len(result_ax.get_lines()) == expected_num_lines, "Should have one line for each group"


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

    assert isinstance(result_ax, Axes), "Result should be an Axes object"
    assert len(result_ax.get_lines()) == expected_num_lines, "Should have only one line"


@pytest.mark.parametrize(
    ("move_legend_outside", "expected_title"),
    [
        (True, "Test Plot Legend Outside"),
        (False, "Test Plot Legend Inside"),
    ],
)
@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_legend_positioning(sample_dataframe, move_legend_outside, expected_title):
    """Test the plot function legend positioning (inside vs outside)."""
    result_ax = line.plot(
        df=sample_dataframe,
        value_col="y",
        x_label="X Axis",
        y_label="Y Axis",
        title=expected_title,
        x_col="x",
        group_col="group",
        move_legend_outside=move_legend_outside,
    )

    # Assert that standard_graph_styles was called with the correct move_legend_outside value
    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title=expected_title,
        x_label="X Axis",
        y_label="Y Axis",
        legend_title=None,
        move_legend_outside=move_legend_outside,
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


@pytest.mark.parametrize(
    ("group_col", "expected_legend", "plot_title"),
    [
        (None, False, "Test Single Line Plot"),
        ("group", True, "Test Grouped Line Plot"),
    ],
)
@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_line_plot_calls_dataframe_plot(mocker, sample_dataframe, group_col, expected_legend, plot_title):
    """Test that pandas.DataFrame.plot is called with correct arguments for single vs grouped plots."""
    # Mock DataFrame's plot method
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    # Call the line plot function
    line.plot(
        df=sample_dataframe,
        value_col="y",
        group_col=group_col,
        x_col="x",
        title=plot_title,
    )

    # Check that DataFrame.plot was called with the correct arguments
    # Since we added highlight functionality, we expect additional parameters
    mock_df_plot.assert_called_once_with(
        ax=mocker.ANY,
        linewidth=3,
        color=mocker.ANY,  # Dynamic color generation
        alpha=1.0,  # New parameter added for highlight functionality
        legend=expected_legend,
        zorder=2,  # New parameter added for highlight functionality
    )


@pytest.mark.parametrize(
    ("value_col", "group_col", "expected_error"),
    [
        (None, None, "value_col is required when df is a DataFrame"),
        (["y", "y1"], "group", "Cannot use both a list for `value_col` and a `group_col`. Choose one."),
    ],
)
def test_dataframe_error_conditions(sample_dataframe, value_col, group_col, expected_error):
    """Test DataFrame error conditions for invalid parameter combinations."""
    if value_col is not None:
        sample_dataframe["y1"] = range(10, 20)

    with pytest.raises(ValueError, match=expected_error):
        line.plot(
            df=sample_dataframe,
            value_col=value_col,
            x_col="x",
            group_col=group_col,
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

    assert isinstance(result_ax, Axes), "Result should be an Axes object"
    expected_num_lines = 2  # One for each store
    assert len(result_ax.get_lines()) == expected_num_lines, "Should have one line for each store"

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

    assert isinstance(result_ax, Axes), "Result should be an Axes object"
    expected_num_lines = 2  # One for each store
    assert len(result_ax.get_lines()) == expected_num_lines, "Should have one line for each store"

    # Get the plotted data for each store
    line_data = {}
    for plot_line in result_ax.get_lines():
        label = plot_line.get_label()
        y_data = plot_line.get_ydata()
        line_data[label] = y_data

    north_data = line_data["Store_North"]
    south_data = line_data["Store_South"]

    # When fill_na_value=None, matplotlib handles missing data with masked arrays
    assert isinstance(north_data, np.ma.MaskedArray), "Store_North data should be a masked array due to missing data"
    assert north_data.mask.any(), "Store_North should have masked values for missing week 4"

    # Store_South with complete data should be a regular numpy array (no masking needed)
    assert isinstance(south_data, np.ndarray), "Store_South data should be a numpy array"
    assert not isinstance(south_data, np.ma.MaskedArray), "Store_South should not be a masked array (has complete data)"


@pytest.mark.parametrize(
    ("x_label", "y_label"),
    [
        ("Categories", "Revenue ($)"),
        ("Time Period", "Sales"),
        (None, None),
    ],
)
def test_plot_series_with_styling_options(sample_series, x_label, y_label):
    """Test that styling options (x_label, y_label) work with Series input."""
    result_ax = line.plot(
        df=sample_series,
        value_col=None,
        x_label=x_label,
        y_label=y_label,
    )

    # Verify it returns an Axes object
    assert isinstance(result_ax, Axes), "Result should be an Axes object"

    # Verify the x-axis data uses the Series index
    lines = result_ax.get_lines()
    plotted_x_data = lines[0].get_xdata()
    # For string indices, matplotlib converts them to numeric positions
    if sample_series.index.dtype == "object":
        assert len(plotted_x_data) == len(sample_series.index), "X-axis should have same length as Series index"
    else:
        assert list(plotted_x_data) == list(range(len(sample_series))), "X-axis should match Series index positions"


@pytest.mark.parametrize(
    ("value_col", "x_col", "group_col", "expected_error"),
    [
        ("revenue", None, None, "When df is a pd.Series, value_col must be None"),
        (None, "revenue", None, "When df is a pd.Series, x_col must be None"),
        (None, None, "revenue", "When df is a pd.Series, group_col must be None"),
    ],
)
def test_invalid_series_parameter_combinations(sample_series, value_col, x_col, group_col, expected_error):
    """Test that invalid parameter combinations with Series raise appropriate ValueError messages."""
    with pytest.raises(ValueError, match=expected_error):
        line.plot(df=sample_series, value_col=value_col, x_col=x_col, group_col=group_col)


@pytest.mark.parametrize(
    ("data", "index", "name"),
    [
        ([12450.50, 15890.25, 18234.75], ["Store_North", "Store_South", "Store_East"], "revenue"),
        ([145, 189, 234, 178], [1, 2, 3, 4], "daily_transactions"),
        ([1245.50, 1567.25, 1834.75], pd.date_range("2024-01-01", periods=3), "daily_sales"),
        ([523, 678, 845, 912, 1034], range(5), "customer_count"),
        ([1, np.nan, 3, np.nan, 5], [1, 2, 3, 4, 5], "with_nans"),  # Series with NaN values
        ([np.nan, np.nan, np.nan], [1, 2, 3], "all_nan"),  # All NaN values
        ([1, 2, 3, 4, 5], [1, 1, 2, 2, 3], "duplicate_index"),  # Duplicate index values
        ([0], [1], "single_zero"),  # Single value Series with zero
    ],
)
def test_series_with_various_configurations(data, index, name):
    """Test Series plotting with various data types, indices, and naming configurations including edge cases."""
    test_series = pd.Series(data, index=index, name=name)
    result_ax = line.plot(df=test_series, value_col=None, title="Series Configuration Test")

    assert isinstance(result_ax, Axes), "Result should be an Axes object"

    # Verify exactly one line is plotted
    lines = result_ax.get_lines()
    assert len(lines) == 1, "Should have exactly one line for Series"

    # For edge cases, handle different validation approaches
    plotted_data = lines[0].get_ydata()

    if name == "all_nan":
        # All NaN Series - matplotlib handles NaN values with masked arrays
        assert len(plotted_data) == len(data), f"Expected {len(data)} data points, got {len(plotted_data)}"
        # Check that all values are masked (NaN)
        if hasattr(plotted_data, "mask"):
            assert plotted_data.mask.all(), "All NaN Series should have all values masked"
    elif name == "with_nans":
        # Series with some NaN values - matplotlib handles with masked arrays
        assert len(plotted_data) == len(data), f"Expected {len(data)} data points, got {len(plotted_data)}"
    else:
        # Regular cases - verify the plotted data matches the input data
        expected_data = data if isinstance(data, list) else list(data)
        actual_data = list(plotted_data)
        assert actual_data == expected_data, f"Plotted data {actual_data} should match input data {expected_data}"

    # Verify data length always matches
    assert len(plotted_data) == len(data), f"Expected {len(data)} data points, got {len(plotted_data)}"


@pytest.fixture
def multi_category_dataframe():
    """A sample dataframe with multiple categories for highlight testing."""
    data = {
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "category": [
            "Electronics",
            "Electronics",
            "Electronics",
            "Clothing",
            "Clothing",
            "Clothing",
            "Home",
            "Home",
            "Home",
        ],
        "revenue": [100, 120, 140, 80, 85, 90, 60, 65, 70],
    }
    return pd.DataFrame(data)


@pytest.fixture
def multi_metric_dataframe():
    """A sample dataframe with multiple metrics as columns for highlight testing."""
    data = {
        "day": range(1, 6),
        "revenue": [100, 110, 120, 130, 140],
        "units_sold": [50, 55, 60, 65, 70],
        "avg_order_value": [2.0, 2.0, 2.0, 2.0, 2.0],
        "profit_margin": [0.2, 0.22, 0.24, 0.26, 0.28],
    }
    return pd.DataFrame(data)


class TestHighlightFeature:
    """Tests for the highlight parameter functionality."""

    @pytest.mark.parametrize(
        ("highlight", "expected_highlighted_count", "expected_context_count"),
        [
            ("Electronics", 1, 2),
            (["Electronics", "Clothing"], 2, 1),
        ],
    )
    def test_highlight_with_group_col(
        self,
        multi_category_dataframe,
        highlight,
        expected_highlighted_count,
        expected_context_count,
    ):
        """Test highlighting with group_col using single string or list of strings."""
        expected_categories_count = 3

        result_ax = line.plot(
            df=multi_category_dataframe,
            x_col="month",
            value_col="revenue",
            group_col="category",
            highlight=highlight,
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"
        lines = result_ax.get_lines()
        assert len(lines) == expected_categories_count, (
            f"Expected {expected_categories_count} lines but got {len(lines)}"
        )

        # Normalize highlight to list for uniform processing
        highlight_list = [highlight] if isinstance(highlight, str) else highlight

        # Check that highlighted lines have higher z-order
        highlighted_lines = []
        context_lines = []
        for plot_line in lines:
            if plot_line.get_label() in highlight_list:
                highlighted_lines.append(plot_line)
            else:
                context_lines.append(plot_line)

        assert len(highlighted_lines) == expected_highlighted_count, (
            f"Expected {expected_highlighted_count} highlighted lines but got {len(highlighted_lines)}"
        )
        assert len(context_lines) == expected_context_count, (
            f"Expected {expected_context_count} context lines but got {len(context_lines)}"
        )

        # Highlighted lines should have higher z-order than context
        for highlighted_line in highlighted_lines:
            for context_line in context_lines:
                assert highlighted_line.get_zorder() > context_line.get_zorder(), (
                    "Highlighted lines should have higher z-order than context lines"
                )

    def test_highlight_with_value_col_list(self, multi_metric_dataframe):
        """Test highlighting with list of value_col."""
        expected_metrics_count = 4
        expected_highlighted_pairs = 2

        result_ax = line.plot(
            df=multi_metric_dataframe,
            x_col="day",
            value_col=["revenue", "units_sold", "avg_order_value", "profit_margin"],
            highlight=["revenue", "profit_margin"],
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"
        lines = result_ax.get_lines()
        assert len(lines) == expected_metrics_count, f"Expected {expected_metrics_count} lines but got {len(lines)}"

        # Check that highlighted lines have higher z-order
        highlighted_lines = []
        context_lines = []
        for plot_line in lines:
            if plot_line.get_label() in ["revenue", "profit_margin"]:
                highlighted_lines.append(plot_line)
            else:
                context_lines.append(plot_line)

        assert len(highlighted_lines) == expected_highlighted_pairs, (
            f"Expected {expected_highlighted_pairs} highlighted lines but got {len(highlighted_lines)}"
        )
        assert len(context_lines) == expected_highlighted_pairs, (
            f"Expected {expected_highlighted_pairs} context lines but got {len(context_lines)}"
        )

        # Highlighted lines should have higher z-order than context
        for highlighted_line in highlighted_lines:
            for context_line in context_lines:
                assert highlighted_line.get_zorder() > context_line.get_zorder(), (
                    "Highlighted lines should have higher z-order than context lines"
                )

    def test_highlight_single_line_raises_error(self, sample_dataframe):
        """Test that highlight raises error for single-line plots."""
        with pytest.raises(ValueError, match="highlight parameter cannot be used with single-line plots"):
            line.plot(df=sample_dataframe, value_col="y", x_col="x", highlight="something")

    def test_highlight_invalid_values_raises_error(self, multi_category_dataframe):
        """Test that highlight raises error for invalid values."""
        with pytest.raises(ValueError, match="highlight values .* not found in available columns"):
            line.plot(
                df=multi_category_dataframe,
                x_col="month",
                value_col="revenue",
                group_col="category",
                highlight=["NonExistent", "AlsoNotThere"],
            )

    def test_highlight_none_behaves_normally(self, multi_category_dataframe):
        """Test that highlight=None behaves like the original implementation."""
        expected_categories_count = 3
        highlighted_linewidth = 3
        highlighted_alpha = 1.0
        highlighted_zorder = 2

        result_ax = line.plot(
            df=multi_category_dataframe,
            x_col="month",
            value_col="revenue",
            group_col="category",
            highlight=None,
        )

        lines = result_ax.get_lines()
        assert len(lines) == expected_categories_count, (
            f"Expected {expected_categories_count} lines but got {len(lines)}"
        )

        # All lines should have highlighted styling (no context lines)
        for plot_line in lines:
            assert plot_line.get_linewidth() == highlighted_linewidth, (
                "All lines should have highlighted linewidth when highlight=None"
            )
            assert plot_line.get_alpha() == highlighted_alpha, (
                "All lines should have highlighted alpha when highlight=None"
            )
            assert plot_line.get_zorder() == highlighted_zorder, (
                "All lines should have highlighted z-order when highlight=None"
            )

    def test_highlight_styling_properties(self, multi_category_dataframe):
        """Test that highlight applies correct styling properties."""
        highlighted_linewidth = 3
        highlighted_alpha = 1.0
        highlighted_zorder = 2
        context_linewidth = 1.5
        context_alpha = 0.6
        context_zorder = 1

        result_ax = line.plot(
            df=multi_category_dataframe,
            x_col="month",
            value_col="revenue",
            group_col="category",
            highlight="Electronics",
        )

        lines = result_ax.get_lines()

        # Find highlighted and context lines
        highlighted_line = None
        context_lines = []
        for plot_line in lines:
            if plot_line.get_label() == "Electronics":
                highlighted_line = plot_line
            else:
                context_lines.append(plot_line)

        # Check styling properties
        assert highlighted_line.get_linewidth() == highlighted_linewidth, (
            "Highlighted line should have default highlighted linewidth"
        )
        assert highlighted_line.get_alpha() == highlighted_alpha, "Highlighted line should have highlighted alpha"
        assert highlighted_line.get_zorder() == highlighted_zorder, "Highlighted line should have highlighted z-order"

        for context_line in context_lines:
            assert context_line.get_linewidth() == context_linewidth, "Context lines should have context linewidth"
            assert context_line.get_alpha() == context_alpha, "Context lines should have context alpha"
            assert context_line.get_zorder() == context_zorder, "Context lines should have context z-order"

    def test_edge_case_empty_highlighted_cols(self, multi_category_dataframe, mocker):
        """Test edge case where highlighted_cols becomes empty (defensive programming)."""
        mocker.patch(
            "pyretailscience.plots.line._categorize_columns",
            return_value=([], ["Electronics", "Clothing", "Home"]),
        )

        result_ax = line.plot(
            df=multi_category_dataframe,
            x_col="month",
            value_col="revenue",
            group_col="category",
            highlight="Electronics",
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"
        lines = result_ax.get_lines()
        assert len(lines) > 0, f"Expected at least one line to be plotted but got {len(lines)}"
