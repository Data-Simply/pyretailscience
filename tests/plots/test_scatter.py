"""Tests for the plots.scatter module."""

from itertools import cycle

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import scatter
from pyretailscience.style import graph_utils as gu

PERIODS = 6
RNG = np.random.default_rng(42)


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
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi-color maps."""
    single_color_gen = cycle(["#FF0000"])
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])

    mocker.patch("pyretailscience.style.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.style.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture
def _mock_gu_functions(mocker):
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


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
