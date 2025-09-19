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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_missing_value_col():
    """Test price architecture plot when value_col doesn't exist."""
    df = pd.DataFrame(
        {
            "price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(KeyError, match="value_col 'unit_price' not found in DataFrame"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_missing_group_col():
    """Test price architecture plot when group_col doesn't exist."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "store": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(KeyError, match="group_col 'retailer' not found in DataFrame"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_non_numeric_value_col():
    """Test price architecture plot with non-numeric value column."""
    df = pd.DataFrame(
        {
            "unit_price": ["low", "medium", "high"],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(ValueError, match="value_col 'unit_price' must be numeric for binning"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=5,
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_invalid_bins_zero():
    """Test price architecture plot with zero bins."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(ValueError, match="bins must be a positive integer"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=0,
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_invalid_bins_negative():
    """Test price architecture plot with negative bins."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(ValueError, match="bins must be a positive integer"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=-5,
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_invalid_bins_list_too_short():
    """Test price architecture plot with bins list too short."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(ValueError, match="bins list must contain at least 2 values"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=[1],
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_invalid_bins_list_non_numeric():
    """Test price architecture plot with non-numeric bins list."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(ValueError, match="All values in bins list must be numeric"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=[1, "invalid", 3],
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_unsorted_bins_list(mocker):
    """Test price architecture plot passes sorted bins to pd.cut."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

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
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[3, 1, 2],  # Unsorted bins
    )

    # Verify pd.cut was called with sorted bins
    mock_cut.assert_called_once()
    call_args = mock_cut.call_args
    assert call_args[1]["bins"] == [1, 2, 3]  # Should be sorted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_invalid_bins_type():
    """Test price architecture plot with invalid bins type."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3],
            "retailer": ["Walmart", "Target", "Amazon"],
        },
    )

    with pytest.raises(TypeError, match="bins must be either an integer or a list of numeric values"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins="invalid",
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_missing_values(mocker):
    """Test price architecture plot handles missing values by dropping rows with NaN."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, None, 4, 5],
            "retailer": ["Walmart", "Walmart", "Target", "Target", None],
        },
    )

    # Expected data after dropping missing values (rows 2 and 4 should be dropped)
    expected_clean_df = pd.DataFrame(
        {
            "unit_price": [1, 2, 4],
            "retailer": ["Walmart", "Walmart", "Target"],
        },
    )

    # Mock dropna to verify it's called and return expected clean data
    mock_dropna = mocker.patch.object(pd.DataFrame, "dropna")
    mock_dropna.return_value = expected_clean_df

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
    )

    # Verify dropna was called on the subset of columns
    mock_dropna.assert_called_once()

    # Verify the function still works with cleaned data
    assert isinstance(result_ax, Axes)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
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
    ("bins", "title"),
    [
        (3, "Test Price Architecture Plot"),
        ([1, 3, 5, 7], "Test Price Architecture Plot with Custom Bins"),
    ],
)
def test_plot_with_bins(simple_price_dataframe, bins, title):
    """Test price architecture plot with integer bins and custom bin boundaries."""
    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=bins,
        title=title,
    )

    assert isinstance(result_ax, Axes)

    expected_retailers = 3  # Walmart, Target, Amazon retailers
    assert len(result_ax.get_xticks()) == expected_retailers

    expected_bins = 3  # 3 price bins/boundaries
    assert len(result_ax.get_yticks()) == expected_bins


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_handles_zero_percentages():
    """Test that plot handles edge case where all percentages might be zero."""
    # Create a DataFrame where all products fall into the same bin
    # This creates a scenario where some bins have 0%
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 1.0, 1.0, 1.0],  # All same price
            "retailer": ["Walmart", "Walmart", "Target", "Target"],
        },
    )

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=5,  # More bins than data points creates empty bins
    )

    # Should not raise division by zero error
    assert isinstance(result_ax, Axes)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_raises_error_when_no_data_in_bins():
    """Test that plot raises error when no data falls within the specified bins."""
    df = pd.DataFrame(
        {
            "unit_price": [5.0, 6.0, 7.0, 8.0],
            "retailer": ["Walmart", "Walmart", "Target", "Target"],
        },
    )

    # Create bins that don't include any of the data points
    with pytest.raises(ValueError, match="All percentages are zero - no data falls within the specified bins"):
        price.plot(
            df=df,
            value_col="unit_price",
            group_col="retailer",
            bins=[1.0, 2.0, 3.0, 4.0],  # All data is above these bins
        )


def test_percentages_sum_to_100_for_each_group():
    """Test that percentages calculated for each group sum to 100%."""
    tolerance = 0.001  # Floating point comparison tolerance

    # Create test data with known distribution
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # 8 items
            "retailer": ["Walmart", "Walmart", "Walmart", "Walmart", "Target", "Target", "Target", "Target"],  # 4 each
        },
    )

    # Clean data like the plot function does
    df_clean = df[["unit_price", "retailer"]].dropna().copy()
    bins = [1.0, 3.0, 5.0, 7.0, 9.0]  # 4 bins

    # Replicate the percentage calculation logic from the plot function
    df_clean["price_bin"] = pd.cut(df_clean["unit_price"], bins=bins, include_lowest=True)
    group_totals = df_clean.groupby("retailer", observed=True).size()
    bin_counts = df_clean.groupby(["retailer", "price_bin"], observed=True).size().unstack(fill_value=0)
    percentages = bin_counts.div(group_totals, axis=0) * 100

    # Verify each group's percentages sum to 100%
    for group in percentages.index:
        group_percentage_sum = percentages.loc[group].sum()
        assert abs(group_percentage_sum - 100.0) < tolerance, (
            f"Group {group} percentages sum to {group_percentage_sum}, not 100%"
        )


def test_individual_percentage_calculations_are_correct():
    """Test that individual percentage calculations are mathematically correct."""
    tolerance = 0.001  # Floating point comparison tolerance

    # Create test data with known, predictable distribution
    df = pd.DataFrame(
        {
            "unit_price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10 items total
            "retailer": ["Walmart"] * 4 + ["Target"] * 6,  # Walmart: 4 items, Target: 6 items
        },
    )

    # Clean data like the plot function does
    df_clean = df[["unit_price", "retailer"]].dropna().copy()
    bins = [0.0, 3.0, 6.0, 11.0]  # 3 bins: [0-3], (3-6], (6-11]

    # Replicate the percentage calculation logic from the plot function
    df_clean["price_bin"] = pd.cut(df_clean["unit_price"], bins=bins, include_lowest=True)
    group_totals = df_clean.groupby("retailer", observed=True).size()
    bin_counts = df_clean.groupby(["retailer", "price_bin"], observed=True).size().unstack(fill_value=0)
    percentages = bin_counts.div(group_totals, axis=0) * 100

    # Expected distribution:
    # Walmart (4 items): [1,2,3,4] -> bin1: [1,2,3] (3 items = 75%), bin2: [4] (1 item = 25%), bin3: 0%
    # Target (6 items): [5,6,7,8,9,10] -> bin1: 0%, bin2: [5,6] (2 items = 33.33%), bin3: [7,8,9,10] (4 items = 66.67%)

    # Verify Walmart percentages
    walmart_percentages = percentages.loc["Walmart"]
    assert abs(walmart_percentages.iloc[0] - 75.0) < tolerance, (
        f"Walmart bin1 should be 75%, got {walmart_percentages.iloc[0]}"
    )
    assert abs(walmart_percentages.iloc[1] - 25.0) < tolerance, (
        f"Walmart bin2 should be 25%, got {walmart_percentages.iloc[1]}"
    )
    assert abs(walmart_percentages.iloc[2] - 0.0) < tolerance, (
        f"Walmart bin3 should be 0%, got {walmart_percentages.iloc[2]}"
    )

    # Verify Target percentages
    target_percentages = percentages.loc["Target"]
    assert abs(target_percentages.iloc[0] - 0.0) < tolerance, (
        f"Target bin1 should be 0%, got {target_percentages.iloc[0]}"
    )
    assert abs(target_percentages.iloc[1] - 33.333333333333336) < tolerance, (
        f"Target bin2 should be ~33.33%, got {target_percentages.iloc[1]}"
    )
    assert abs(target_percentages.iloc[2] - 66.66666666666667) < tolerance, (
        f"Target bin3 should be ~66.67%, got {target_percentages.iloc[2]}"
    )


def test_percentage_calculations_edge_cases():
    """Test percentage calculations with edge cases like single items and uneven distributions."""
    tolerance = 0.001  # Floating point comparison tolerance

    # Test case 1: Single item per group
    df_single = pd.DataFrame(
        {
            "unit_price": [2.0, 8.0],
            "retailer": ["Walmart", "Target"],
        },
    )

    # Clean and calculate
    df_clean = df_single[["unit_price", "retailer"]].dropna().copy()
    bins = [0.0, 5.0, 10.0]  # 2 bins
    df_clean["price_bin"] = pd.cut(df_clean["unit_price"], bins=bins, include_lowest=True)
    group_totals = df_clean.groupby("retailer", observed=True).size()
    bin_counts = df_clean.groupby(["retailer", "price_bin"], observed=True).size().unstack(fill_value=0)
    percentages = bin_counts.div(group_totals, axis=0) * 100

    # Each group has 1 item, so one bin should be 100%, other should be 0%
    walmart_percentages = percentages.loc["Walmart"]
    target_percentages = percentages.loc["Target"]

    # Walmart (price=2.0) falls in first bin [0-5]
    assert abs(walmart_percentages.iloc[0] - 100.0) < tolerance, "Walmart should have 100% in first bin"
    assert abs(walmart_percentages.iloc[1] - 0.0) < tolerance, "Walmart should have 0% in second bin"

    # Target (price=8.0) falls in second bin (5-10]
    assert abs(target_percentages.iloc[0] - 0.0) < tolerance, "Target should have 0% in first bin"
    assert abs(target_percentages.iloc[1] - 100.0) < tolerance, "Target should have 100% in second bin"

    # Test case 2: Highly uneven distribution
    df_uneven = pd.DataFrame(
        {
            "unit_price": [1.0] * 9 + [9.0],  # 9 items at 1.0, 1 item at 9.0
            "retailer": ["Walmart"] * 10,
        },
    )

    # Clean and calculate
    df_clean = df_uneven[["unit_price", "retailer"]].dropna().copy()
    bins = [0.0, 5.0, 10.0]  # 2 bins
    df_clean["price_bin"] = pd.cut(df_clean["unit_price"], bins=bins, include_lowest=True)
    group_totals = df_clean.groupby("retailer", observed=True).size()
    bin_counts = df_clean.groupby(["retailer", "price_bin"], observed=True).size().unstack(fill_value=0)
    percentages = bin_counts.div(group_totals, axis=0) * 100

    # Should have 90% in first bin, 10% in second bin
    walmart_percentages = percentages.loc["Walmart"]
    assert abs(walmart_percentages.iloc[0] - 90.0) < tolerance, (
        f"Should be 90% in first bin, got {walmart_percentages.iloc[0]}"
    )
    assert abs(walmart_percentages.iloc[1] - 10.0) < tolerance, (
        f"Should be 10% in second bin, got {walmart_percentages.iloc[1]}"
    )
