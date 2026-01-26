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

    def teardown_method(self):
        """Clean up after each test method."""
        plt.close("all")

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
            index_col="region",
            value_to_index="North",
            include_only_groups=["A", "B"],
        )

        assert isinstance(result_ax, plt.Axes)

        # Verify that only the filtered groups appear in the plot
        y_labels = [label.get_text() for label in result_ax.get_yticklabels()]
        plotted_groups = set(y_labels)
        expected_groups = {"A"}

        assert plotted_groups == expected_groups, (
            f"Found groups {plotted_groups - expected_groups} that should have been filtered out. "
            f"Expected only groups from {expected_groups}."
        )

        assert len(plotted_groups) > 0, "No groups were plotted - filtering may have removed all data."

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
        ytick_labels = [label.get_text() for label in result_ax.get_yticklabels()]
        xtick_labels = [label.get_text() for label in result_ax.get_xticklabels()]
        assert not any(
            label == "" or label is None or str(label).lower() == "nan" or str(label).lower() == "na"
            for label in ytick_labels
        )
        assert not any(
            label == "" or label is None or str(label).lower() == "nan" or str(label).lower() == "na"
            for label in xtick_labels
        )
        bar_values = [patch.get_width() for patch in result_ax.patches]
        assert not any(pd.isna(val) for val in bar_values)

    def test_nan_index_values_present_in_data(self, test_data):
        """Test that NaN index values are present in the index DataFrame when expected."""
        df = test_data.copy()
        df.loc[df["category"] == "A", "sales"] = np.nan

        index_df = get_indexes(
            df,
            value_to_index="A",
            index_col="category",
            value_col="sales",
            group_col="category",
        )
        assert index_df["index"].isna().any(), "Expected at least one NaN value in the 'index' column"

    def test_filter_by_index_values(self):
        """Test that the function can filter indexes by value."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C", "D", "E"] * 10,
                "sales": [100, 200, 50, 300, 75] * 10,
            },
        )

        # Calculate what the actual index values will be so we can choose a meaningful threshold
        index_df = get_indexes(
            df=df,
            value_to_index="A",
            index_col="category",
            value_col="sales",
            group_col="category",
            offset=100,
        )

        # Choose a threshold that will actually filter some data
        sorted_indexes = index_df.sort_values("index")["index"].tolist()

        threshold = sorted_indexes[0] - 0.1

        expected_above_threshold_groups = index_df[index_df["index"] > threshold]["category"].tolist()

        result_ax = plot(
            df,
            value_col="sales",
            group_col="category",
            index_col="category",
            value_to_index="A",
            filter_above=threshold,
        )

        filtered_labels = [t.get_text() for t in result_ax.get_yticklabels()]

        assert len(filtered_labels) == len(expected_above_threshold_groups), (
            f"Expected {len(expected_above_threshold_groups)} groups, got {len(filtered_labels)}"
        )
        assert all(label in expected_above_threshold_groups for label in filtered_labels), (
            f"Expected groups {expected_above_threshold_groups}, but got {filtered_labels}"
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

        # Sort to find expected top and bottom groups
        expected_top_group = result_df.nlargest(1, "index")["category"].iloc[0]
        expected_bottom_group = result_df.nsmallest(1, "index")["category"].iloc[0]

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
        # For top_n=1 case
        labels = [t.get_text() for t in result_ax.get_yticklabels()]
        assert len(labels) == 1, f"Expected 1 group for top_n=1, got {len(labels)}"
        assert labels[0] == expected_top_group, f"Expected top group '{expected_top_group}', got '{labels[0]}'"

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

        # For bottom_n=1 case
        labels = [t.get_text() for t in result_ax.get_yticklabels()]
        assert len(labels) == 1, f"Expected 1 group for bottom_n=1, got {len(labels)}"
        assert labels[0] == expected_bottom_group, f"Expected bottom group '{expected_bottom_group}', got '{labels[0]}'"

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
            # For combined top_n=1 and bottom_n=1 case
            labels = [t.get_text() for t in result_ax.get_yticklabels()]
            expected_len = 2
            assert len(labels) == expected_len, f"Expected 2 groups for top_n=1 and bottom_n=1, got {len(labels)}"
            assert expected_top_group in labels, f"Expected top group '{expected_top_group}' not found in {labels}"
            assert expected_bottom_group in labels, (
                f"Expected bottom group '{expected_bottom_group}' not found in {labels}"
            )

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

    @pytest.mark.parametrize(
        ("sort_order", "expected_pairs", "expected_y_labels"),
        [
            (
                "ascending",
                [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y"), ("C", "X"), ("C", "Y")],
                ["A", "B", "C"],
            ),
            (
                "descending",
                [("C", "Y"), ("C", "X"), ("B", "Y"), ("B", "X"), ("A", "Y"), ("A", "X")],
                ["C", "B", "A"],
            ),
        ],
    )
    def test_sort_and_plot_with_series_col(
        self,
        sort_order,
        expected_pairs,
        expected_y_labels,
    ):
        """Combined test: validates sorting of dataframe and sorting in plot output."""
        test_df = pd.DataFrame(
            {
                "category": ["A", "B", "C", "A", "B", "C"],
                "region": ["X", "X", "X", "Y", "Y", "Y"],
                "sales": [100, 200, 150, 120, 180, 160],
                "baseline_category": ["A", "A", "A", "A", "A", "A"],
            },
        )
        ascending_flag = sort_order == "ascending"

        sorted_df = test_df.sort_values(by=["category", "region"], ascending=ascending_flag)
        actual_pairs = list(zip(sorted_df["category"], sorted_df["region"], strict=False))
        assert actual_pairs == expected_pairs, (
            f"{sort_order=} sort mismatch: expected {expected_pairs}, got {actual_pairs}"
        )

        ax = plot(
            test_df,
            value_col="sales",
            group_col="category",
            index_col="baseline_category",
            value_to_index="A",  # Compare all categories against baseline category A
            series_col="region",
            sort_by="group",
            sort_order=sort_order,
        )

        assert isinstance(ax, plt.Axes)

        # Verify y-axis labels reflect sorted categories
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == expected_y_labels, (
            f"{sort_order=} y-ticks mismatch: expected {expected_y_labels}, got {y_labels}"
        )

        # Verify legend contains series_col values
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        expected_legend_labels = ["X", "Y"]
        assert set(legend_labels) == set(expected_legend_labels), (
            f"Legend mismatch: expected {expected_legend_labels}, got {legend_labels}"
        )

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
