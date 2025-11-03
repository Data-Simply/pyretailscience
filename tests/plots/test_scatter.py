"""Tests for the plots.scatter module."""

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import scatter
from pyretailscience.plots.styles import graph_utils as gu

PERIODS = 6
RNG = np.random.default_rng(42)
NBR_CANDIDATES_DEFAULT = 50  # Default number of candidates for textalloc


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_sales_dataframe():
    """A sample dataframe for Sales, Profit, and Expenses data."""
    data = {
        "date": pd.date_range("2023-01-01", periods=PERIODS, freq="ME"),
        "sales": RNG.integers(1000, 5000, size=PERIODS),
        "profit": RNG.integers(200, 1000, size=PERIODS),
        "expenses": RNG.integers(500, 3000, size=PERIODS),
        "category": ["Electronics", "Clothing", "Furniture", "Electronics", "Clothing", "Furniture"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_product_dataframe():
    """A sample dataframe for product performance with labels."""
    data = {
        "price": [10.5, 15.0, 8.5, 12.0, 20.0],
        "units_sold": [150, 120, 200, 90, 60],
        "product_name": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
        "category": ["Electronics", "Electronics", "Home", "Home", "Electronics"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_store_dataframe():
    """A sample dataframe for store performance by region with labels."""
    data = {
        "revenue": [50000, 75000, 45000, 90000, 65000, 55000],
        "customer_count": [1200, 1800, 1000, 2200, 1500, 1300],
        "store_id": ["S001", "S002", "S003", "S004", "S005", "S006"],
        "region": ["North", "North", "South", "South", "East", "East"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi-color maps."""
    single_color_gen = cycle(["#FF0000"])
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])

    mocker.patch("pyretailscience.plots.styles.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.plots.styles.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture
def _mock_gu_functions(mocker):
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_column(sample_sales_dataframe):
    """Test scatter plot with a single value column."""
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Monthly Sales Trend",
        x_col="date",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_group_col(sample_sales_dataframe):
    """Test scatter plot with a group column."""
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Sales by Category",
        x_col="date",
        group_col="category",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_columns(sample_sales_dataframe):
    """Test scatter plot with multiple value columns."""
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col=["sales", "profit"],
        x_label="Date",
        y_label="Amount",
        title="Sales & Profit Trends",
        x_col="date",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_columns_with_group_col(sample_sales_dataframe):
    """Test scatter plot when using multiple columns along with a group column."""
    with pytest.raises(ValueError, match="Cannot use both a list for `value_col` and a `group_col`. Choose one."):
        scatter.plot(
            df=sample_sales_dataframe,
            value_col=["sales", "profit", "expenses"],
            x_label="Date",
            y_label="Amount",
            title="Sales, Profit & Expenses",
            x_col="date",
            group_col="category",
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(sample_sales_dataframe):
    """Test scatter plot adds source text."""
    source_text = "Source: Test Data"
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Test Plot Source Text",
        x_col="date",
        source_text=source_text,
    )
    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_column_series(sample_sales_dataframe):
    """Test scatter plot with a single value as a Pandas Series."""
    result_ax = scatter.plot(
        df=sample_sales_dataframe["sales"],
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Sales Trend (Series)",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


def test_plot_with_labels_single_series(sample_product_dataframe, mocker):
    """Test scatter plot with labels on single series."""
    mock_textalloc = mocker.patch("pyretailscience.plots.scatter.ta.allocate")

    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        x_col="price",
        label_col="product_name",
        title="Product Performance",
        x_label="Price",
        y_label="Units Sold",
    )

    # Verify basic plot structure
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == "Product Performance"
    assert result_ax.get_xlabel() == "Price"
    assert result_ax.get_ylabel() == "Units Sold"

    # Verify scatter plot was created with correct data points
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) >= 1, "No scatter plot collections found"
    offsets = collections[0].get_offsets()
    assert len(offsets) == len(sample_product_dataframe)

    # Check that textalloc was called with correct parameters
    mock_textalloc.assert_called_once()
    args, kwargs = mock_textalloc.call_args

    ax_arg = args[0]
    x_coords = args[1]
    y_coords = args[2]
    labels = args[3]

    assert ax_arg == result_ax
    assert len(x_coords) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} x coordinates, got {len(x_coords)}"
    )
    assert len(y_coords) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} y coordinates, got {len(y_coords)}"
    )
    assert len(labels) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} labels, got {len(labels)}"
    )

    # Verify labels match expected product names
    expected_labels = sample_product_dataframe["product_name"].tolist()
    expected_prices = sample_product_dataframe["price"].tolist()
    expected_units = sample_product_dataframe["units_sold"].tolist()

    assert set(labels) == set(expected_labels), f"Labels don't match: expected {expected_labels}, got {labels}"

    # Verify coordinates match data points
    assert list(x_coords) == expected_prices, (
        f"X coordinates don't match: expected {expected_prices}, got {list(x_coords)}"
    )
    assert list(y_coords) == expected_units, (
        f"Y coordinates don't match: expected {expected_units}, got {list(y_coords)}"
    )


def test_plot_with_labels_grouped_series(sample_store_dataframe, mocker):
    """Test scatter plot with labels on grouped series."""
    mock_textalloc = mocker.patch("pyretailscience.plots.scatter.ta.allocate")

    result_ax = scatter.plot(
        df=sample_store_dataframe,
        value_col="revenue",
        x_col="customer_count",
        group_col="region",
        label_col="store_id",
        title="Store Performance by Region",
        x_label="Customer Count",
        y_label="Revenue",
    )

    # Verify basic plot structure
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == "Store Performance by Region"
    assert result_ax.get_xlabel() == "Customer Count"
    assert result_ax.get_ylabel() == "Revenue"

    # Verify scatter plot collections were created for groups
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) >= 1, "No scatter plot collections found"

    # For grouped scatter plots, each group creates its own collection
    # We expect one collection per unique region
    unique_regions = sample_store_dataframe["region"].nunique()
    assert len(collections) == unique_regions, (
        f"Expected {unique_regions} collections for {unique_regions} regions, got {len(collections)}"
    )

    # Check that textalloc was called with correct parameters
    mock_textalloc.assert_called_once()
    args, kwargs = mock_textalloc.call_args

    ax_arg = args[0]
    x_coords = args[1]
    y_coords = args[2]
    labels = args[3]

    assert ax_arg == result_ax
    assert len(x_coords) == len(sample_store_dataframe), (
        f"Expected {len(sample_store_dataframe)} x coordinates, got {len(x_coords)}"
    )
    assert len(y_coords) == len(sample_store_dataframe), (
        f"Expected {len(sample_store_dataframe)} y coordinates, got {len(y_coords)}"
    )
    assert len(labels) == len(sample_store_dataframe), (
        f"Expected {len(sample_store_dataframe)} labels, got {len(labels)}"
    )

    # Verify labels contain correct store IDs
    expected_store_ids = set(sample_store_dataframe["store_id"].tolist())
    actual_labels = set(labels)
    assert actual_labels == expected_store_ids, (
        f"Store ID labels don't match: expected {expected_store_ids}, got {actual_labels}"
    )


def test_plot_with_labels_custom_kwargs(sample_product_dataframe, mocker):
    """Test scatter plot with custom label_kwargs passed to textalloc."""
    mock_textalloc = mocker.patch("pyretailscience.plots.scatter.ta.allocate")

    custom_kwargs = {
        "nbr_candidates": 100,
        "min_distance": 0.1,
    }

    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        x_col="price",
        label_col="product_name",
        label_kwargs=custom_kwargs,
        title="Product Performance with Custom Labels",
    )

    # Verify basic plot structure
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == "Product Performance with Custom Labels"

    # Verify scatter plot was created
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) >= 1, "No scatter plot collections found"

    # Verify textalloc was called with custom kwargs
    mock_textalloc.assert_called_once()
    args, kwargs = mock_textalloc.call_args

    ax_arg = args[0]
    x_coords = args[1]
    y_coords = args[2]
    labels = args[3]

    # Verify coordinates and labels were created
    assert len(x_coords) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} x coordinates, got {len(x_coords)}"
    )
    assert len(y_coords) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} y coordinates, got {len(y_coords)}"
    )
    assert len(labels) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} labels, got {len(labels)}"
    )

    # Check that custom kwargs were passed through to textalloc
    assert kwargs["nbr_candidates"] == custom_kwargs["nbr_candidates"], (
        f"nbr_candidates not passed correctly: expected {custom_kwargs['nbr_candidates']}, got {kwargs.get('nbr_candidates')}"
    )
    assert kwargs["min_distance"] == custom_kwargs["min_distance"], (
        f"min_distance not passed correctly: expected {custom_kwargs['min_distance']}, got {kwargs.get('min_distance')}"
    )
    assert ax_arg == result_ax

    # Verify labels contain correct product names
    expected_labels = set(sample_product_dataframe["product_name"].tolist())
    actual_labels = set(labels)
    assert actual_labels == expected_labels, f"Labels don't match: expected {expected_labels}, got {actual_labels}"


@pytest.mark.parametrize(
    ("value_col", "label_col", "expected_error", "expected_msg"),
    [
        (["units_sold", "price"], "product_name", ValueError, "label_col is not supported when value_col is a list"),
        ("units_sold", "nonexistent_col", KeyError, "label_col 'nonexistent_col' not found in DataFrame"),
    ],
)
def test_plot_label_validation_errors(sample_product_dataframe, value_col, label_col, expected_error, expected_msg):
    """Test various label validation errors."""
    with pytest.raises(expected_error, match=expected_msg):
        scatter.plot(
            df=sample_product_dataframe,
            value_col=value_col,
            x_col="price",
            label_col=label_col,
        )


def test_plot_labels_with_nan_values(sample_product_dataframe, mocker):
    """Test scatter plot with labels when data contains NaN values."""
    mock_textalloc = mocker.patch("pyretailscience.plots.scatter.ta.allocate")

    # Add some NaN values to test filtering
    df_with_nan = sample_product_dataframe.copy()
    df_with_nan.loc[1, "units_sold"] = np.nan
    df_with_nan.loc[3, "product_name"] = np.nan

    result_ax = scatter.plot(
        df=df_with_nan,
        value_col="units_sold",
        x_col="price",
        label_col="product_name",
        title="Product Performance with NaN values",
    )

    assert isinstance(result_ax, Axes)
    mock_textalloc.assert_called_once()
    args, kwargs = mock_textalloc.call_args
    labels = args[3]
    # Should have fewer texts due to NaN filtering
    expected_non_nan_count = len(df_with_nan.dropna(subset=["units_sold", "product_name"]))
    assert len(labels) == expected_non_nan_count


def test_plot_labels_using_index_as_x(sample_product_dataframe, mocker):
    """Test scatter plot with labels using index as x-axis."""
    mock_textalloc = mocker.patch("pyretailscience.plots.scatter.ta.allocate")

    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        label_col="product_name",
        title="Product Performance using Index",
    )

    # Verify basic plot structure
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == "Product Performance using Index"

    # Verify scatter plot was created
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) >= 1, "No scatter plot collections found"

    # Verify data points match input series length
    offsets = collections[0].get_offsets()
    assert len(offsets) == len(sample_product_dataframe), (
        f"Expected {len(sample_product_dataframe)} points, got {len(offsets)}"
    )

    mock_textalloc.assert_called_once()
    args, kwargs = mock_textalloc.call_args
    labels = args[3]
    assert len(labels) == len(sample_product_dataframe)


def test_plot_without_labels_no_textalloc_called(sample_product_dataframe, mocker):
    """Test that textalloc is not called when no labels are specified."""
    mock_textalloc = mocker.patch("pyretailscience.plots.scatter.ta.allocate")

    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        x_col="price",
        title="Product Performance without Labels",
    )

    assert isinstance(result_ax, Axes)
    # textalloc should not have been called
    mock_textalloc.assert_not_called()
