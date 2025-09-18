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
    """Test price architecture plot automatically sorts unsorted bins list."""
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


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_integer_bins(simple_price_dataframe):
    """Test price architecture plot with integer bins."""
    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
        title="Test Price Architecture Plot",
    )

    assert isinstance(result_ax, Axes)

    expected_retailers = 3  # Walmart, Target, Amazon retailers
    assert len(result_ax.get_xticks()) == expected_retailers

    expected_bins = 3  # 3 price bins/boundaries
    assert len(result_ax.get_yticks()) == expected_bins


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_list_bins(simple_price_dataframe):
    """Test price architecture plot with custom bin boundaries."""
    result_ax = price.plot(
        df=simple_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=[1, 3, 5, 7],
        title="Test Price Architecture Plot with Custom Bins",
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
def test_plot_single_group_uses_single_color():
    """Test that single group data successfully creates a plot (implying single color logic works)."""
    df = pd.DataFrame(
        {
            "unit_price": [1, 2, 3, 4],
            "retailer": ["Walmart", "Walmart", "Walmart", "Walmart"],  # Only one retailer
        },
    )

    # Test that single group data works without errors
    # The fact it succeeds implies the single color mapping logic is working
    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=2,
    )

    assert isinstance(result_ax, Axes)

    # Verify we have 1 group's data plotted
    expected_groups = 1  # Only Walmart
    assert len(result_ax.get_xticks()) == expected_groups


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_many_groups_uses_multi_color(sample_price_dataframe):
    """Test that many groups data successfully creates a plot (implying multi-color logic works)."""
    result_ax = price.plot(
        df=sample_price_dataframe,
        value_col="unit_price",
        group_col="retailer",
        bins=3,
    )

    assert isinstance(result_ax, Axes)

    # Verify we have multiple groups plotted (4+ groups trigger multi-color mapping)
    expected_groups = 4  # Walmart, Target, Amazon, Best Buy (threshold is 4)
    assert len(result_ax.get_xticks()) == expected_groups


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
