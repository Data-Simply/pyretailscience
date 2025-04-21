"""Tests for the plots.area module."""

from itertools import cycle

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import area
from pyretailscience.style import graph_utils as gu

PERIODS = 6
RNG = np.random.default_rng(42)


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for Jeans sales data."""
    data = {
        "transaction_date": np.repeat(pd.date_range("2023-01-01", periods=PERIODS, freq="ME"), 3),
        "unit_spend": RNG.integers(1, 6, size=3 * PERIODS),
        "category": ["Jeans", "Shoes", "Dresses"] * PERIODS,
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
def test_plot_single_column(sample_dataframe):
    """Test the plot function with a single value column."""
    result_ax = area.plot(
        df=sample_dataframe,
        value_col="unit_spend",
        x_label="Transaction Date",
        y_label="Sales",
        title="Jeans Sales (Single Column)",
        x_col="transaction_date",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_group_col(sample_dataframe):
    """Test the plot function with a group column (stacked area chart)."""
    result_ax = area.plot(
        df=sample_dataframe,
        value_col="unit_spend",
        x_label="Transaction Date",
        y_label="Sales",
        title="Sales by Category",
        x_col="transaction_date",
        group_col="category",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_columns(sample_dataframe):
    """Test the plot function with multiple columns as a stacked area chart."""
    sample_dataframe["additional_spend"] = RNG.integers(1, 6, size=3 * PERIODS)
    result_ax = area.plot(
        df=sample_dataframe,
        value_col=["unit_spend", "additional_spend"],
        x_label="Transaction Date",
        y_label="Sales",
        title="Sales by Product Category",
        x_col="transaction_date",
        alpha=0.9,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_columns_with_group_col(sample_dataframe):
    """Test the plot function when using multiple columns along with a group column."""
    sample_dataframe["additional_spend"] = RNG.integers(1, 6, size=3 * PERIODS)
    with pytest.raises(ValueError, match="Cannot use both a list for `value_col` and a `group_col`. Choose one."):
        area.plot(
            df=sample_dataframe,
            value_col=["unit_spend", "additional_spend", "Shoes"],
            x_label="Transaction Date",
            y_label="Sales",
            title="Sales by Product Category",
            x_col="transaction_date",
            group_col="category",
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(sample_dataframe):
    """Test the plot function adds source text to the plot."""
    source_text = "Source: Test Data"

    result_ax = area.plot(
        df=sample_dataframe,
        value_col="unit_spend",
        x_label="X Axis",
        y_label="Y Axis",
        title="Test Plot Source Text",
        x_col="transaction_date",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_column_series(sample_dataframe):
    """Test the plot function with a single value as a Pandas Series."""
    result_ax = area.plot(
        df=sample_dataframe["unit_spend"],
        value_col="unit_spend",
        x_label="Transaction Date",
        y_label="Sales",
        title="Product Sales (Series)",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0
