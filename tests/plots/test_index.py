"""Tests for the index plot module."""

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from pyretailscience.plots.index import (
    BASELINE_INDEX,
    filter_by_groups,
    filter_by_value_thresholds,
    filter_top_bottom_n,
    get_indexes,
    plot,
)
from pyretailscience.plots.styles.colors import get_named_color

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


class TestGetIndexesZeroDivision:
    """Tests for get_indexes division by zero edge cases.

    Verifies that when denominators are zero (overall total, subset total,
    or group proportion), the code returns NaN instead of raising errors.
    """

    @pytest.mark.parametrize(
        "df_data",
        [
            pytest.param(
                {
                    "region": ["North", "South"],
                    "category": ["Electronics", "Electronics"],
                    "sales": [0, 0],
                },
                id="zero_overall_total",
            ),
            pytest.param(
                {
                    "region": ["North", "South", "North", "South"],
                    "category": ["Electronics", "Electronics", "Grocery", "Grocery"],
                    "sales": [0, 0, 100, 200],
                },
                id="zero_subset_total",
            ),
        ],
    )
    def test_returns_all_nan_when_totals_are_zero(self, df_data):
        """Test that get_indexes produces all NaN index values when totals are zero."""
        df = pd.DataFrame(df_data)

        result = get_indexes(
            df,
            value_to_index="Electronics",
            index_col="category",
            value_col="sales",
            group_col="region",
        )
        assert result["index"].isna().all(), "Expected all NaN index values when totals are zero"

    def test_zero_overall_total_with_subgroup_returns_nan(self):
        """Test that get_indexes produces NaN index when a subgroup's overall total is zero."""
        df = pd.DataFrame(
            {
                "store": ["Mall", "Mall", "Mall", "Mall", "Outlet", "Outlet", "Outlet", "Outlet"],
                "region": ["North", "South", "North", "South", "North", "South", "North", "South"],
                "category": [
                    "Electronics",
                    "Electronics",
                    "Grocery",
                    "Grocery",
                    "Electronics",
                    "Electronics",
                    "Grocery",
                    "Grocery",
                ],
                "sales": [10, 20, 30, 40, 0, 0, 0, 0],
            },
        )

        result = get_indexes(
            df,
            value_to_index="Electronics",
            index_col="category",
            value_col="sales",
            group_col="region",
            index_subgroup_col="store",
        )
        mall_rows = result[result["store"] == "Mall"]
        assert not mall_rows["index"].isna().any(), "Expected valid index for subgroup with non-zero overall total"

        outlet_rows = result[result["store"] == "Outlet"]
        assert outlet_rows["index"].isna().all(), "Expected NaN index for subgroup with zero overall total"

    def test_zero_group_proportion_returns_nan(self):
        """Test that get_indexes produces NaN for a group with zero overall proportion."""
        df = pd.DataFrame(
            {
                "region": ["North", "South", "East"],
                "category": ["Electronics", "Electronics", "Electronics"],
                "sales": [100, 200, 0],
            },
        )

        result = get_indexes(
            df,
            value_to_index="Electronics",
            index_col="category",
            value_col="sales",
            group_col="region",
        )
        east_row = result[result["region"] == "East"]
        assert east_row["index"].isna().all(), "Expected NaN index for group with zero proportion_overall"

        non_zero_rows = result[result["region"] != "East"]
        assert not non_zero_rows["index"].isna().any(), "Expected valid index for groups with non-zero proportions"


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
        assert result_ax.get_xlim()[0] < BASELINE_INDEX < result_ax.get_xlim()[1]

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
            offset=BASELINE_INDEX,
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

    @pytest.mark.parametrize(
        ("sort_by", "sort_order", "top_n", "bottom_n", "expected_y_labels"),
        [
            pytest.param(
                "value",
                "ascending",
                3,
                2,
                ["Snacks", "Meat", "Dairy", "Bakery", "Produce"],
                id="value_asc_top3_bot2",
            ),
            pytest.param(
                "value",
                "descending",
                3,
                2,
                ["Produce", "Bakery", "Dairy", "Meat", "Snacks"],
                id="value_desc_top3_bot2",
            ),
            pytest.param("value", "ascending", 3, None, ["Dairy", "Bakery", "Produce"], id="value_asc_top3_only"),
            pytest.param("value", "ascending", None, 2, ["Snacks", "Meat"], id="value_asc_bot2_only"),
            pytest.param(
                "value",
                "ascending",
                None,
                None,
                ["Snacks", "Meat", "Dairy", "Bakery", "Produce"],
                id="value_asc_no_filter",
            ),
            pytest.param(
                "group",
                "ascending",
                3,
                2,
                ["Bakery", "Dairy", "Meat", "Produce", "Snacks"],
                id="group_asc_top3_bot2",
            ),
            pytest.param(
                "group",
                "descending",
                None,
                None,
                ["Snacks", "Produce", "Meat", "Dairy", "Bakery"],
                id="group_desc_no_filter",
            ),
            pytest.param(
                None,
                "ascending",
                None,
                None,
                ["Bakery", "Dairy", "Meat", "Produce", "Snacks"],
                id="no_sort_asc",
            ),
            pytest.param(
                None,
                "descending",
                None,
                None,
                ["Bakery", "Dairy", "Meat", "Produce", "Snacks"],
                id="no_sort_desc",
            ),
            pytest.param(None, "ascending", 2, 2, ["Produce", "Bakery", "Meat", "Snacks"], id="no_sort_top2_bot2"),
        ],
    )
    def test_plot_sort_order(self, sort_by, sort_order, top_n, bottom_n, expected_y_labels):
        """Test that y-axis labels reflect the correct sort order with optional top_n/bottom_n filtering."""
        # Index values (ascending): Snacks(-37.9), Meat(-25.5), Dairy(-21.0), Bakery(24.1), Produce(30.3)
        df = pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
                "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
                "spend": [100, 200, 150, 300, 50, 120, 80, 200, 100, 90],
            },
        )

        ax = plot(
            df,
            value_col="spend",
            group_col="department",
            index_col="cust_type",
            value_to_index="Loyalty",
            sort_by=sort_by,
            sort_order=sort_order,
            top_n=top_n,
            bottom_n=bottom_n,
        )

        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == expected_y_labels

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


class TestColorByThreshold:
    """Tests for color_by_threshold functionality in the index plot."""

    def teardown_method(self):
        """Clean up after each test method."""
        plt.close("all")

    @pytest.fixture
    def threshold_data(self):
        """Return data with index values that fall above, below, and between default thresholds.

        Computed visual index values:
            Dairy ≈ 99 (neutral), Bakery ≈ 120 (neutral/positive boundary),
            Meat ≈ 72 (negative), Produce ≈ 126 (positive), Snacks ≈ 60 (negative).
        This ensures all three color branches (positive, negative, neutral) are exercised
        for both the default (80, 120) and custom (90, 110) highlight ranges.
        """
        return pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
                "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
                "spend": [130, 200, 150, 300, 50, 90, 80, 200, 100, 90],
            },
        )

    @pytest.mark.parametrize(
        ("expected_color_names", "plot_kwargs"),
        [
            pytest.param(
                # Default range (80, 120), sorted by value ascending:
                # Snacks≈60 (neg), Meat≈72 (neg), Dairy≈99 (neutral), Bakery≈120 (neutral), Produce≈126 (pos)
                ["negative", "negative", "neutral", "neutral", "positive"],
                {"sort_by": "value", "sort_order": "ascending"},
                id="default_range_value_sort",
            ),
            pytest.param(
                # Custom range (90, 110), sorted by value ascending:
                # Snacks≈60 (neg), Meat≈72 (neg), Dairy≈99 (neutral), Bakery≈120 (pos), Produce≈126 (pos)
                ["negative", "negative", "neutral", "positive", "positive"],
                {"highlight_range": (90, 110), "sort_by": "value", "sort_order": "ascending"},
                id="custom_range_value_sort",
            ),
            pytest.param(
                # Default range (80, 120), sorted by group ascending:
                # Bakery≈120 (neutral), Dairy≈99 (neutral), Meat≈72 (neg), Produce≈126 (pos), Snacks≈60 (neg)
                ["neutral", "neutral", "negative", "positive", "negative"],
                {},
                id="default_range_group_sort",
            ),
        ],
    )
    def test_bars_colored_by_threshold(self, threshold_data, expected_color_names, plot_kwargs):
        """Test that bars are colored positive/negative/neutral based on highlight range thresholds."""
        ax = plot(
            threshold_data,
            value_col="spend",
            group_col="department",
            index_col="cust_type",
            value_to_index="Loyalty",
            color_by_threshold=True,
            **plot_kwargs,
        )

        expected_colors = [get_named_color(name) for name in expected_color_names]

        # Filter to only bar patches (alpha=1.0), excluding axvspan highlight (alpha=0.1)
        bar_patches = [p for p in ax.patches if p.get_alpha() is None or p.get_alpha() == 1.0]
        actual_colors = [to_hex(p.get_facecolor()) for p in bar_patches]
        assert actual_colors == expected_colors

    def test_raises_error_when_highlight_range_is_none(self, threshold_data):
        """Test that ValueError is raised when color_by_threshold=True but highlight_range is None."""
        with pytest.raises(ValueError, match="color_by_threshold requires highlight_range to be set"):
            plot(
                threshold_data,
                value_col="spend",
                group_col="department",
                index_col="cust_type",
                value_to_index="Loyalty",
                highlight_range=None,
                color_by_threshold=True,
            )

    def test_raises_error_when_series_col_provided(self, threshold_data):
        """Test that ValueError is raised when color_by_threshold=True with series_col."""
        with pytest.raises(ValueError, match="color_by_threshold cannot be used when series_col is provided"):
            plot(
                threshold_data,
                value_col="spend",
                group_col="department",
                index_col="cust_type",
                value_to_index="Loyalty",
                series_col="cust_type",
                color_by_threshold=True,
            )
