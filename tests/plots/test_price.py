"""Tests for the price architecture bubble plot module."""

from itertools import cycle

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pyretailscience.plots import price
from pyretailscience.plots.styles import graph_utils as gu


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_price_dataframe():
    """A sample dataframe for testing price architecture plots."""
    data = {
        "product_id": range(1, 101),
        "unit_price": [
            # Walmart - mix of low and medium prices
            *np.random.default_rng(42).uniform(1, 3, 25),
            # Target - mostly medium prices
            *np.random.default_rng(42).uniform(2, 5, 25),
            # Amazon - higher prices
            *np.random.default_rng(42).uniform(4, 8, 25),
            # Best Buy - wide range
            *np.random.default_rng(42).uniform(1, 10, 25),
        ],
        "retailer": (["Walmart"] * 25 + ["Target"] * 25 + ["Amazon"] * 25 + ["Best Buy"] * 25),
        "country": (["US"] * 50 + ["UK"] * 50),
    }
    return pd.DataFrame(data)


@pytest.fixture
def simple_price_dataframe():
    """A simple dataframe for predictable testing."""
    data = {
        "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "retailer": ["Walmart", "Walmart", "Target", "Target", "Amazon", "Amazon"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi color maps."""
    single_color_gen = cycle(["#FF0000"])  # Mocked single-color generator (e.g., red)
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF", "#FFFF00"])  # Mocked multi-color generator
    mocker.patch("pyretailscience.plots.styles.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.plots.styles.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mock the standard graph utilities functions."""
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


def test_plot_with_empty_dataframe():
    """Test price architecture plot with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["unit_price", "retailer"])

    with pytest.raises(ValueError, match="Cannot plot with empty DataFrame"):
        price.plot(
            df=empty_df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


def test_plot_missing_value_col(simple_price_dataframe):
    """Test price architecture plot when value_col doesn't exist."""
    # Remove the unit_price column to test missing value_col
    df = simple_price_dataframe.drop(columns=["unit_price"])

    with pytest.raises(KeyError, match="value_col 'unit_price' not found in DataFrame"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


def test_plot_missing_group_col(simple_price_dataframe):
    """Test price architecture plot when group_col doesn't exist."""
    # Remove the retailer column to test missing group_col
    df = simple_price_dataframe.drop(columns=["retailer"])

    with pytest.raises(KeyError, match="group_col 'retailer' not found in DataFrame"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


def test_plot_non_numeric_value_col(simple_price_dataframe):
    """Test price architecture plot with non-numeric value column."""
    # Replace unit_price with non-numeric values to test validation
    df = simple_price_dataframe.copy()
    df["unit_price"] = ["low", "medium", "high", "low", "medium", "high"]

    with pytest.raises(ValueError, match="value_col 'unit_price' must be numeric for binning"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


@pytest.mark.parametrize("bins", [0, -5])
def test_plot_invalid_bins_non_positive(bins, simple_price_dataframe):
    """Test price architecture plot with zero or negative bins."""
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=bins,
        )


def test_plot_invalid_bins_list_too_short(simple_price_dataframe):
    """Test price architecture plot with bins list too short."""
    with pytest.raises(ValueError, match="bins list must contain at least 2 values"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=[1],
        )


def test_plot_invalid_bins_list_non_numeric(simple_price_dataframe):
    """Test price architecture plot with non-numeric bins list."""
    with pytest.raises(ValueError, match="All values in bins list must be numeric"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=[1, "invalid", 3],
        )


def test_plot_with_unsorted_bins_list(mocker, simple_price_dataframe):
    """Test price architecture plot passes sorted bins to pd.cut."""
    # Mock pd.cut to capture the bins parameter and return realistic intervals
    mock_cut = mocker.patch("pandas.cut")
    # Create mock intervals that mimic pandas.cut behavior
    intervals = [
        pd.Interval(left=1, right=2, closed="right"),
        pd.Interval(left=2, right=3, closed="right"),
        pd.Interval(left=2, right=3, closed="right"),
    ]
    mock_cut.return_value = pd.Series(intervals, name="price_bin")

    price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=[3, 1, 2],  # Unsorted bins
    )

    # Verify pd.cut was called with sorted bins
    mock_cut.assert_called_once()
    call_args = mock_cut.call_args
    assert call_args[1]["bins"] == [1, 2, 3]  # Should be sorted


def test_plot_invalid_bins_type(simple_price_dataframe):
    """Test price architecture plot with invalid bins type."""
    with pytest.raises(TypeError, match="bins must be either an integer or a list of numeric values"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins="invalid",
        )


def test_plot_with_missing_values(mocker):
    """Test price architecture plot handles missing values by dropping rows with NaN."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, None, 4, 5],
            "retailer": ["Walmart", "Walmart", "Target", "Target", None],
        },
    )

    # Mock pd.cut to capture the cleaned data that gets passed to it
    mock_cut = mocker.patch("pandas.cut")
    # Create mock intervals that mimic pandas.cut behavior
    intervals = [
        pd.Interval(left=1, right=2, closed="right"),
        pd.Interval(left=1, right=2, closed="right"),
        pd.Interval(left=2, right=4, closed="right"),
    ]
    mock_cut.return_value = pd.Series(intervals, name="price_bin")

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
    )

    # Verify pd.cut was called and extract the cleaned data
    mock_cut.assert_called_once()
    cleaned_data = mock_cut.call_args[0][0]  # First positional argument to pd.cut

    # Verify that missing values were properly dropped
    # Expected: rows with indices 0, 1, 3 (original data [1, 2, 4])
    # Row 2 (unit_price=None) and row 4 (retailer=None) should be dropped
    expected_values = [1.0, 2.0, 4.0]

    assert list(cleaned_data.values) == expected_values, f"Expected {expected_values}, got {list(cleaned_data.values)}"

    # Also verify the retailer data was cleaned correctly by checking the DataFrame index
    # The cleaned DataFrame should have the correct corresponding retailer values
    expected_bins = 3
    # The function should have properly cleaned both columns together
    assert len(cleaned_data) == expected_bins, f"Expected 3 clean rows, got {len(cleaned_data)}"

    # Verify the function still works with cleaned data
    assert isinstance(result_ax, Axes)


def test_plot_all_missing_values():
    """Test price architecture plot with all missing values."""
    df = pd.DataFrame(
        {
            "unit_price": [None, None, None],
            "retailer": [None, None, None],
        },
    )

    with pytest.raises(ValueError, match="value_col 'unit_price' must be numeric for binning"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=3,
        )


@pytest.mark.parametrize(
    ("bins"),
    [
        (3),
        ([1, 3, 5, 7]),
    ],
)
def test_plot_with_bins(simple_price_dataframe, bins):
    """Test price architecture plot with integer bins and custom bin boundaries."""
    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=bins,
    )

    assert isinstance(result_ax, Axes)

    expected_retailers = 3  # Walmart, Target, Amazon retailers
    assert len(result_ax.get_xticks()) == expected_retailers

    expected_bins = 3  # 3 price bins/boundaries
    assert len(result_ax.get_yticks()) == expected_bins


def test_plot_basic_functionality(sample_price_dataframe, mocker):
    """Test basic price architecture plot functionality with labels and titles."""
    # Mock standard_graph_styles to capture title and label parameters
    mock_standard_styles = mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles")
    mock_standard_styles.side_effect = lambda ax, **kwargs: ax

    result_ax = price.plot(
        df=sample_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=5,
        title="Price Distribution Analysis",
        x_label="Retailers",
        y_label="Price Bands",
    )

    assert isinstance(result_ax, Axes)

    # Verify that standard_graph_styles was called with correct parameters
    mock_standard_styles.assert_called_once()
    call_kwargs = mock_standard_styles.call_args[1]
    assert call_kwargs["title"] == "Price Distribution Analysis"
    assert call_kwargs["x_label"] == "Retailers"
    assert call_kwargs["y_label"] == "Price Bands"

    # Verify we have the expected number of retailers and bins
    expected_retailers = 4  # Walmart, Target, Amazon, Best Buy
    assert len(result_ax.get_xticks()) == expected_retailers

    expected_bins = 5
    assert len(result_ax.get_yticks()) == expected_bins


def test_plot_with_country_grouping(sample_price_dataframe):
    """Test price architecture plot with country grouping."""
    result_ax = price.plot(
        df=sample_price_dataframe,
        value_col="unit_price",
        group_col="country",
        bins=4,
        title="Price Distribution by Country",
    )

    assert isinstance(result_ax, Axes)

    expected_countries = 2  # US, UK
    assert len(result_ax.get_xticks()) == expected_countries


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_calls_standard_styling(simple_price_dataframe):
    """Test that standard graph styling functions are called."""
    title = "Test Title"
    x_label = "Test X Label"
    y_label = "Test Y Label"
    legend_title = "Test Legend"

    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=True,
    )

    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=None,
        move_legend_outside=False,
        show_legend=False,
    )

    gu.standard_tick_styles.assert_called_once_with(ax=result_ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(simple_price_dataframe):
    """Test that source text is added when provided."""
    source_text = "Source: Test Data"

    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


def test_plot_with_kwargs(simple_price_dataframe, mocker):
    """Test that additional kwargs are passed to scatter plot."""
    # Mock ax.scatter to capture kwargs
    mock_scatter = mocker.patch("matplotlib.axes.Axes.scatter")

    price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        alpha=0.8,
        s=200,
        marker="^",
    )

    # Verify scatter was called with the custom kwargs
    mock_scatter.assert_called()

    # Check that our custom kwargs were passed through
    # Note: alpha and s are handled specially, but marker should pass through
    call_kwargs = mock_scatter.call_args[1]
    assert "marker" in call_kwargs
    assert call_kwargs["marker"] == "^"


def test_plot_raises_error_when_no_data_in_bins(simple_price_dataframe):
    """Test that plot raises error when no data falls within the specified bins."""
    # simple_price_dataframe has prices [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # so bins [7.0, 8.0, 9.0, 10.0] will exclude all data
    with pytest.raises(ValueError, match="All proportions are zero - no data falls within the specified bins"):
        price.plot(
            df=simple_price_dataframe,
            value_col="unit_price",
            group_col="retailer",
            bins=[7.0, 8.0, 9.0, 10.0],  # All data is below these bins
        )


def test_percentages_sum_to_100_for_each_group(mocker):
    """Test that percentages calculated by the plot function for each group sum to 100%."""
    # Create test data with known distribution
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # 8 items
            "retailer": ["Walmart", "Walmart", "Walmart", "Walmart", "Target", "Target", "Target", "Target"],  # 4 each
        },
    )

    # Mock the scatter plot to capture the actual size values (percentages) being plotted
    mock_scatter = mocker.patch("matplotlib.axes.Axes.scatter")

    # Set a known scale factor for testing
    scale_factor = 1000

    # Call the actual plot function with known scale factor
    price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[1.0, 3.0, 5.0, 7.0, 9.0],  # 4 bins
        s=scale_factor,
    )

    # Extract the actual data passed to scatter
    mock_scatter.assert_called_once()
    scatter_call = mock_scatter.call_args
    sizes = scatter_call.kwargs["s"]
    x_positions = scatter_call[0][0]  # retailer positions

    # Group sizes by retailer position
    walmart_sizes = [sizes[i] for i, x in enumerate(x_positions) if x == 0]  # x=0 is Walmart
    target_sizes = [sizes[i] for i, x in enumerate(x_positions) if x == 1]  # x=1 is Target

    # Test that bubble sizes for each group sum to 1.0 (100%) when divided by scale factor
    # With absolute scaling: scaled_size = proportion_value * scale_factor
    # So: sum(scaled_sizes) / scale_factor should equal 1.0 for each group

    walmart_proportion_sum = sum(walmart_sizes) / scale_factor
    target_proportion_sum = sum(target_sizes) / scale_factor

    # Both groups should sum to 1.0 (within floating point tolerance)
    tolerance = 0.001  # 0.1% tolerance for floating point precision
    assert abs(walmart_proportion_sum - 1.0) < tolerance, (
        f"Walmart proportions should sum to 1.0, got {walmart_proportion_sum}"
    )
    assert abs(target_proportion_sum - 1.0) < tolerance, (
        f"Target proportions should sum to 1.0, got {target_proportion_sum}"
    )

    # Verify that all non-zero sizes were plotted (no data should be filtered incorrectly)
    assert len(sizes) > 0, "Should have plotted some bubbles"
    assert all(s > 0 for s in sizes), "All plotted bubbles should have positive size"


def test_individual_percentage_calculations_are_correct(mocker):
    """Test that the plot function calculates exact proportions correctly with known data distribution.

    Includes edge case testing with a third retailer having 100% in one band and 0% in others.
    """
    # Create test data with known, predictable distribution that includes edge cases
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5],  # 9 items total
            "retailer": ["Walmart"] * 4
            + ["Target"] * 4
            + ["Amazon"],  # Walmart: 4 items, Target: 4 items, Amazon: 1 item
        },
    )
    scale_factor = 800  # Default s_scale

    # Mock scatter to capture the actual proportion values being plotted
    mock_scatter = mocker.patch("matplotlib.axes.Axes.scatter")

    # Call the actual plot function with bins that create predictable distributions
    price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[0.0, 2.5, 6.5, 9.0, 10.0],  # 4 bins: [0-2.5], (2.5-6.5], (6.5-9], (9-10]
        s=scale_factor,
    )

    # Extract the actual data passed to scatter
    mock_scatter.assert_called_once()
    scatter_call = mock_scatter.call_args
    x_data = scatter_call[0][0]  # x coordinates (retailers)
    y_data = scatter_call[0][1]  # y coordinates (bins)
    sizes = scatter_call.kwargs["s"]  # sizes (scaled proportions)

    # Expected distribution with bins [0.0, 2.5, 6.5, 9.0, 10.0]:
    # Walmart [1,2,3,4]: bin0=[1,2] (2/4=50%), bin1=[3,4] (2/4=50%), bin2=[] (0%), bin3=[] (0%)
    # Target [5,6,7,8]: bin0=[] (0%), bin1=[5,6] (2/4=50%), bin2=[7,8] (2/4=50%), bin3=[] (0%)
    # Amazon [9.5]: bin0=[] (0%), bin1=[] (0%), bin2=[] (0%), bin3=[9.5] (1/1=100%)

    # Retailer names in alphabetical order (same as pandas groupby)
    # Amazon=0, Target=1, Walmart=2
    expected_data = (
        pd.DataFrame(
            {
                "x": [2, 2, 1, 1, 0],  # Walmart, Walmart, Target, Target, Amazon
                "y": [0, 1, 1, 2, 3],  # bin indices
                "size": np.array([0.5, 0.5, 0.5, 0.5, 1.0]) * scale_factor,
            },
        )
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )

    # Reconstruct actual data from scatter call
    actual_data = (
        pd.DataFrame(
            {
                "x": x_data,
                "y": y_data,
                "size": sizes,
            },
        )
        .sort_values(["x", "y"])
        .reset_index(drop=True)
    )

    # Compare expected vs actual
    pd.testing.assert_frame_equal(actual_data, expected_data)
