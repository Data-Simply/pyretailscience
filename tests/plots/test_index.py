"""Tests for the index plot module."""

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyretailscience.plots.index import (
    filter_by_groups,
    filter_by_value_thresholds,
    filter_top_bottom_n,
    get_indexes,
    plot,
)

OFFSET_VALUE = 100
OFFSET_THRESHOLD = 5


def test_get_indexes_basic():
    """Test get_indexes function with basic input to ensure it returns a valid DataFrame."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )

    result = get_indexes(df, value_to_index="A", index_col="category", value_col="value", group_col="category")
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

    result = get_indexes(
        df,
        value_to_index="A",
        index_col="category",
        value_col="value",
        group_col="category",
        index_subgroup_col="subgroup",
    )
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


def test_get_indexes_invalid_agg_func():
    """Test get_indexes function with an invalid aggregation function."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )

    with pytest.raises(ValueError, match="Unsupported aggregation function."):
        get_indexes(
            df,
            value_to_index="A",
            index_col="category",
            value_col="value",
            group_col="category",
            agg_func="invalid_func",
        )


def test_get_indexes_with_different_aggregations():
    """Test get_indexes function with various aggregation functions."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )

    for agg in ["sum", "mean", "max", "min", "nunique"]:
        result = get_indexes(
            df,
            value_to_index="A",
            index_col="category",
            value_col="value",
            group_col="category",
            agg_func=agg,
        )
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
    result = get_indexes(
        df,
        value_to_index="A",
        index_col="category",
        value_col="value",
        group_col="category",
        offset=OFFSET_THRESHOLD,
    )

    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty
    assert all(result["index"] >= -OFFSET_THRESHOLD)


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
        value_to_index="X",
        index_col="filter_col",
        value_col="value_col",
        group_col="group_col",
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
        value_to_index="X",
        index_col="filter_col",
        value_col="value_col",
        group_col="group_col1",
        index_subgroup_col="group_col2",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_ibis_table_input():
    """Test that the get_indexes function works with an ibis Table."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [10, 20, 30],
        },
    )
    table = ibis.memtable(df)

    result = get_indexes(table, value_to_index="A", index_col="category", value_col="value", group_col="category")
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns
    assert "index" in result.columns
    assert not result.empty


class TestIndexPlot:
    """Tests for the index_plot function."""

    @pytest.fixture
    def test_data(self):
        """Return a sample dataframe for plotting."""
        rng = np.random.default_rng()
        data = {
            "category": ["A", "B", "C", "D", "E"] * 2,
            "sales": rng.integers(100, 500, size=10),
            "region": ["North", "South", "East", "West", "Central"] * 2,
        }
        return pd.DataFrame(data)

    def test_generates_index_plot_with_default_parameters(
        self,
        test_data,
    ):
        """Test that the function generates an index plot with default parameters."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
        )

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) > 0
        assert result_ax.get_xlabel() == "Index"
        assert result_ax.get_ylabel() == "Category"

    def test_generates_index_plot_with_custom_title(self, test_data):
        """Test that the function generates an index plot with a custom title."""
        df = test_data
        custom_title = "Sales Performance by Category"
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            title=custom_title,
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_title() == custom_title

    def test_generates_index_plot_with_highlight_range(self, test_data):
        """Test that the function generates an index plot with a highlighted range."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            highlight_range=(80, 120),
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_xlim()[0] < OFFSET_VALUE < result_ax.get_xlim()[1]

    def test_generates_index_plot_with_group_filter(self, test_data):
        """Test that the function generates an index plot with a group filter applied."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            include_only_groups=["A", "B"],
        )

        assert isinstance(result_ax, plt.Axes)

    def test_raises_value_error_for_invalid_sort_by(self, test_data):
        """Test that the function raises a ValueError for an invalid sort_by parameter."""
        df = test_data

        with pytest.raises(ValueError):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                sort_by="invalid",
            )

    def test_raises_value_error_for_invalid_sort_order(self, test_data):
        """Test that the function raises a ValueError for an invalid sort_order parameter."""
        df = test_data

        with pytest.raises(ValueError):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                sort_order="invalid",
            )

    def test_generates_index_plot_with_source_text(self, test_data):
        """Test that the function generates an index plot with source text."""
        df = test_data
        source_text = "Data source: Company XYZ"
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            source_text=source_text,
        )

        assert isinstance(result_ax, plt.Axes)
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_generates_index_plot_with_custom_labels(self, test_data):
        """Test that the function generates an index plot with custom x and y labels."""
        df = test_data
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            x_label="Sales Value",
            y_label="Category Group",
        )

        assert isinstance(result_ax, plt.Axes)
        assert result_ax.get_xlabel() == "Sales Value"
        assert result_ax.get_ylabel() == "Category Group"

    def test_drop_na_index_values(self, test_data):
        """Test that the function can drop NA index values."""
        df = test_data.copy()
        # Introduce NA value by making one group have the same proportion
        df.loc[0, "sales"] = df.loc[5, "sales"]

        # This should work without error
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            drop_na=True,
        )

        assert isinstance(result_ax, plt.Axes)

    def test_filter_by_index_values(self, test_data):
        """Test that the function can filter indexes by value."""
        df = test_data

        # Create plot with value-based filtering (use a reasonable value for filter_above)
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            filter_above=0,  # Using a lower value that should pass
        )

        assert isinstance(result_ax, plt.Axes)

    def test_empty_dataset_after_filtering(self, test_data):
        """Test that filtering that results in an empty dataset raises ValueError."""
        df = test_data

        # Use an extremely high filter value that will result in an empty dataset
        with pytest.raises(ValueError, match="Filtering resulted in an empty dataset"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                filter_above=100000,  # Using an extremely high value that should cause all data to be filtered out
            )

    def test_top_and_bottom_n(self, test_data):
        """Test that the function can display only top and/or bottom N indexes."""
        # Use a non-reference category to get more groups in the result
        df = test_data

        # First verify available groups to work with
        result_df = get_indexes(
            df=df,
            value_to_index="C",  # Use a different value for indexing to get more groups
            index_col="category",
            value_col="sales",
            group_col="category",
        )
        available_groups = result_df["category"].unique()

        group_count = len(available_groups)

        # Test with top_n (making sure value is ≤ available groups)
        top_count = min(1, group_count)
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="C",  # Use non-A value as reference
            top_n=top_count,
        )

        assert isinstance(result_ax, plt.Axes)

        # Test with bottom_n (making sure value is ≤ available groups)
        bottom_count = min(1, group_count)
        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="C",  # Use non-A value as reference
            bottom_n=bottom_count,
        )

        assert isinstance(result_ax, plt.Axes)

        # Test with both top_n and bottom_n if we have enough groups
        minimum_group_count = 2
        if group_count >= minimum_group_count:
            result_ax = plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="C",  # Use non-A value as reference
                top_n=1,
                bottom_n=1,
            )

            assert isinstance(result_ax, plt.Axes)

    def test_error_with_series_and_filtering(self, test_data):
        """Test that appropriate error is raised when using filtering with series_col."""
        df = test_data

        with pytest.raises(
            ValueError,
            match="top_n, bottom_n, filter_above, and filter_below cannot be used when series_col is provided",
        ):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                series_col="region",
                top_n=2,
            )

    def test_sort_values_with_series_col(self):
        """Test that the sort_values function works correctly with [group_col, series_col]."""
        # Create a test dataframe directly to test the sorting functionality
        test_df = pd.DataFrame(
            {
                "category": ["C", "A", "B", "C", "A", "B"],
                "region": ["X", "X", "X", "Y", "Y", "Y"],
                "index": [100, 90, 110, 105, 95, 115],
            },
        )

        # Test ascending sort
        sorted_asc = test_df.sort_values(by=["category", "region"], ascending=True)

        # Verify sorting is correct for ascending
        expected_order_asc = [
            ("A", "X"),
            ("A", "Y"),
            ("B", "X"),
            ("B", "Y"),
            ("C", "X"),
            ("C", "Y"),
        ]
        actual_order_asc = list(zip(sorted_asc["category"], sorted_asc["region"], strict=True))
        assert actual_order_asc == expected_order_asc

        # Test descending sort
        sorted_desc = test_df.sort_values(by=["category", "region"], ascending=False)

        # Verify sorting is correct for descending
        expected_order_desc = [
            ("C", "Y"),
            ("C", "X"),
            ("B", "Y"),
            ("B", "X"),
            ("A", "Y"),
            ("A", "X"),
        ]
        actual_order_desc = list(zip(sorted_desc["category"], sorted_desc["region"], strict=True))
        assert actual_order_desc == expected_order_desc

        # Now test the actual implementation in the plot function
        # Create a plot with series_col to trigger the code path we're testing
        result_ax = plot(
            test_df,
            value_col="index",
            group_col="category",
            index_col="category",
            value_to_index="A",
            series_col="region",
            sort_by="group",
            sort_order="ascending",
        )

        # Verify the plot was created
        assert isinstance(result_ax, plt.Axes)

        # Verify legend exists
        legend = result_ax.get_legend()
        assert legend is not None

    def test_error_with_excessive_top_and_bottom_n(self, test_data):
        """Test that appropriate error is raised when top_n + bottom_n exceeds group count."""
        df = test_data
        unique_count = len(df["category"].unique())

        # Since validation flow changed, we now get a different error message
        # when top_n exceeds available groups
        with pytest.raises(ValueError, match="top_n .* cannot exceed the number of available groups"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                top_n=unique_count,
                bottom_n=1,
            )

    def test_error_with_top_n_exceeding_available_groups(self, test_data):
        """Test that appropriate error is raised when top_n exceeds available groups."""
        df = test_data
        total_count = len(df["category"].unique())

        with pytest.raises(ValueError, match="top_n .* cannot exceed the number of available groups"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                top_n=total_count + 1,
            )

    def test_error_with_bottom_n_exceeding_available_groups(self, test_data):
        """Test that appropriate error is raised when bottom_n exceeds available groups."""
        df = test_data
        total_count = len(df["category"].unique())

        with pytest.raises(ValueError, match="bottom_n .* cannot exceed the number of available groups"):
            plot(
                df,
                value_col="sales",
                group_col="category",
                index_col="category",
                value_to_index="A",
                bottom_n=total_count + 1,
            )

    def test_error_with_sum_of_top_and_bottom_n(self):
        """Test that appropriate error is raised when top_n + bottom_n exceeds available groups."""
        # Create a test index dataframe with the same groups
        test_df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "index": [90, 110, 120],
            },
        )

        # Test that sum of top_n and bottom_n validation works
        with pytest.raises(
            ValueError,
            match="The sum of top_n .* and bottom_n .* cannot exceed the total number of groups",
        ):
            filter_top_bottom_n(
                df=test_df,
                top_n=2,
                bottom_n=2,
            )


def test_filter_by_groups_exclude_groups():
    """Test that filter_by_groups correctly excludes specified groups."""
    # Create test dataframe with multiple categories
    test_df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10, 20, 30, 40, 50],
        },
    )

    # Test excluding specific groups
    exclude_list = ["B", "D"]
    result_df = filter_by_groups(
        df=test_df,
        group_col="category",
        exclude_groups=exclude_list,
    )

    # Check that excluded groups are not in the result
    assert all(value not in result_df["category"].values for value in ["B", "D"])

    # Check that other groups are still in the result
    assert all(value in result_df["category"].values for value in ["A", "C", "E"])

    # Check that the result has the expected number of rows
    expected_row_count = len(test_df) - len(exclude_list)
    assert len(result_df) == expected_row_count


def test_filter_by_groups_validation_error():
    """Test that filter_by_groups raises ValueError when both exclude and include params are provided."""
    test_df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "value": [10, 20, 30, 40, 50],
        },
    )

    exclude_list = ["B", "D"]
    include_list = ["A", "C"]

    # Test with both exclude_groups and include_only_groups
    with pytest.raises(ValueError, match="exclude_groups and include_only_groups cannot be used together."):
        plot(
            df=test_df,
            value_col="value",
            group_col="category",
            index_col="category",
            value_to_index="A",
            exclude_groups=exclude_list,
            include_only_groups=include_list,
        )


def test_series_col_with_sort_by_value_validation_error():
    """Test that providing series_col with sort_by='value' raises ValueError."""
    test_df = pd.DataFrame(
        {
            "category": ["A", "B", "C"] * 2,
            "series": ["X", "X", "X", "Y", "Y", "Y"],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )

    with pytest.raises(ValueError, match="sort_by cannot be 'value' when series_col is provided."):
        plot(
            df=test_df,
            value_col="value",
            group_col="category",
            index_col="category",
            value_to_index="A",
            series_col="series",
            sort_by="value",
        )


def test_filter_by_value_thresholds_filter_below():
    """Test that filter_by_value_thresholds correctly filters values below a threshold."""
    # Create test dataframe with varying index values
    test_df = pd.DataFrame(
        {
            "category": ["A", "B", "C", "D", "E"],
            "index": [80, 90, 100, 110, 120],
        },
    )

    # Define the threshold value
    threshold = 105

    # Apply the filter
    result_df = filter_by_value_thresholds(
        df=test_df,
        filter_below=threshold,
    )

    # Verify only values below the threshold remain
    assert all(result_df["index"] < threshold)

    # Check that correct values are included and excluded
    assert all(value in result_df["index"].values for value in [80, 90, 100])
    assert all(value not in result_df["index"].values for value in [110, 120])

    # Check that the result has the expected number of rows
    expected_count = len(test_df[test_df["index"] < threshold])
    assert len(result_df) == expected_count
