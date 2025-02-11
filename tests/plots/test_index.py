"""Tests for the index plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyretailscience.plots.index import get_indexes, plot

OFFSET_VALUE = 100


def test_get_indexes_single_column():
    """Test that the function works with a single column index."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [77.77777778, 100, 106.0606]})
    output = get_indexes(
        df=df,
        index_col="group_col",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_two_columns():
    """Test that the function works with two columns as the index."""
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
            "group_col2": ["D", "D", "D", "D", "D", "D", "E", "E", "E", "E", "E", "E"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col2": ["D", "D", "D", "E", "E", "E"],
            "group_col1": ["A", "B", "C", "A", "B", "C"],
            "index": [77.77777778, 100, 106.0606, 98.51851852, 100, 100.9661836],
        },
    )
    output = get_indexes(
        df=df,
        index_col="group_col1",
        index_subgroup_col="group_col2",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_offset():
    """Test that the function works with an offset parameter."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [-22.22222222, 0, 6.060606061]})
    output = get_indexes(
        df=df,
        index_col="group_col",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
        offset=OFFSET_VALUE,  # Replace magic number with the constant
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_agg_func():
    """Test that the function works with the nunique agg_func parameter."""
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 5, 8],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col1": ["A", "B", "C"],
            "index": [140, 140, 46.6666667],
        },
    )
    output = get_indexes(
        df=df,
        index_col="group_col1",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
        agg_func="nunique",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_index_filter_all_same():
    """Test that the function raises a ValueError when all the values in the index filter are the same."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "X", "X", "X", "X", "X"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    # Assert a value error will be reaised
    with pytest.raises(ValueError):
        get_indexes(
            df=df,
            df_index_filter=[True, True, True, True, True, True],
            index_col="group_col",
            value_col="value_col",
        )

    with pytest.raises(ValueError):
        get_indexes(
            df=df,
            df_index_filter=[False, False, False, False, False, False],
            index_col="group_col",
            value_col="value_col",
        )


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
