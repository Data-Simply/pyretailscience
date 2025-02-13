"""Tests for the index plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyretailscience.plots.index import get_indexes, plot

OFFSET_VALUE = 100
OFFSET_THRESHOLD = -5


def test_get_indexes_basic():
    """Test get_indexes function with basic input to ensure it returns a valid DataFrame."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )
    df_index_filter = [True, False, True, False, True, False]

    result = get_indexes(df, df_index_filter, "category", "value")
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


def test_get_indexes_with_subgroup():
    """Test get_indexes function when a subgroup column is provided."""
    df = pd.DataFrame(
        {
            "subgroup": ["X", "X", "X", "Y", "Y", "Y"],
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )
    df_index_filter = [True, False, True, False, True, False]

    result = get_indexes(df, df_index_filter, "category", "value", index_subgroup_col="subgroup")
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


def test_get_indexes_invalid_filter():
    """Test get_indexes function with an invalid filter where all values are True."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )
    df_index_filter = [True, True, True]  # Invalid case

    with pytest.raises(ValueError, match="The df_index_filter cannot be all True or all False."):
        get_indexes(df, df_index_filter, "category", "value")


def test_get_indexes_invalid_agg_func():
    """Test get_indexes function with an invalid aggregation function."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )
    df_index_filter = [True, False, True]

    with pytest.raises(ValueError, match="Unsupported aggregation function."):
        get_indexes(df, df_index_filter, "category", "value", agg_func="invalid_func")


def test_get_indexes_with_different_aggregations():
    """Test get_indexes function with various aggregation functions."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )
    df_index_filter = [True, False, True, False, True, False]

    for agg in ["sum", "mean", "max", "min", "nunique"]:
        result = get_indexes(df, df_index_filter, "category", "value", agg_func=agg)
        assert isinstance(result, pd.DataFrame)
        assert "category" in result.columns
        assert "index" in result.columns
        assert not result.empty


def test_get_indexes_with_offset():
    """Test get_indexes function with an offset value."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )
    df_index_filter = [True, False, True]
    result = get_indexes(df, df_index_filter, "category", "value", offset=5)

    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty
    assert all(result["index"] >= OFFSET_THRESHOLD)


class TestIndexPlot:
    """Tests for the index_plot function."""

    @pytest.fixture()
    def test_data(self):
        """Return a sample dataframe for plotting."""
        rng = np.random.default_rng()
        data = {
            "category": ["A", "B", "C", "D", "E"] * 2,
            "sales": rng.integers(100, 500, size=10),
            "region": ["North", "South", "East", "West", "Central"] * 2,
        }
        return pd.DataFrame(data)

    @pytest.fixture()
    def df_index_filter(self, test_data):
        """Return a boolean filter for the dataframe."""
        return [True, False, True, True, False, True, True, False, True, False]

    def test_generates_index_plot_with_default_parameters(
        self,
        test_data,
        df_index_filter,
    ):
        """Test that the function generates an index plot with default parameters."""
        df = test_data
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
        )

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) > 0  # Ensures bars are plotted
        assert result_ax.get_xlabel() == "Index"
        assert result_ax.get_ylabel() == "Category"

    def test_generates_index_plot_with_custom_title(self, test_data, df_index_filter):
        """Test that the function generates an index plot with a custom title."""
        df = test_data
        custom_title = "Sales Performance by Category"
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            title=custom_title,
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_title() == custom_title

    def test_generates_index_plot_with_highlight_range(
        self,
        test_data,
        df_index_filter,
    ):
        """Test that the function generates an index plot with a highlighted range."""
        df = test_data
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            highlight_range=(80, 120),
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_xlim()[0] < OFFSET_VALUE < result_ax.get_xlim()[1]

    def test_generates_index_plot_with_group_filter(self, test_data, df_index_filter):
        """Test that the function generates an index plot with a group filter applied."""
        df = test_data
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            include_only_groups=["A", "B"],
        )

        assert isinstance(result_ax, plt.Axes)

    def test_raises_value_error_for_invalid_sort_by(self, test_data, df_index_filter):
        """Test that the function raises a ValueError for an invalid sort_by parameter."""
        df = test_data

        with pytest.raises(ValueError):
            plot(
                df,
                df_index_filter=df_index_filter,
                value_col="sales",
                group_col="category",
                sort_by="invalid",
            )

    def test_raises_value_error_for_invalid_sort_order(
        self,
        test_data,
        df_index_filter,
    ):
        """Test that the function raises a ValueError for an invalid sort_order parameter."""
        df = test_data

        with pytest.raises(ValueError):
            plot(
                df,
                df_index_filter=df_index_filter,
                value_col="sales",
                group_col="category",
                sort_order="invalid",
            )

    def test_generates_index_plot_with_source_text(self, test_data, df_index_filter):
        """Test that the function generates an index plot with source text."""
        df = test_data
        source_text = "Data source: Company XYZ"
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            source_text=source_text,
        )

        assert isinstance(result_ax, plt.Axes)
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_generates_index_plot_with_custom_labels(self, test_data, df_index_filter):
        """Test that the function generates an index plot with custom x and y labels."""
        df = test_data
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            x_label="Index Value",
            y_label="Product Category",
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_xlabel() == "Index Value"
        assert result_ax.get_ylabel() == "Product Category"

    def test_generates_index_plot_with_legend(self, test_data, df_index_filter):
        """Test that the function generates an index plot with a legend when series_col is provided."""
        df = test_data
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            series_col="region",
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_legend() is not None
        assert len(result_ax.get_legend().get_texts()) > 0

    def test_generates_index_plot_without_legend(self, test_data, df_index_filter):
        """Test that the function generates an index plot without a legend when series_col is not provided."""
        df = test_data
        result_ax = plot(
            df,
            df_index_filter=df_index_filter,
            value_col="sales",
            group_col="category",
            series_col=None,
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_legend() is None
