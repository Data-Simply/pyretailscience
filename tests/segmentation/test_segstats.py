"""Tests for the SegTransactionStats class."""

import ibis
import numpy as np
import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper, get_option, option_context
from pyretailscience.segmentation.segstats import SegTransactionStats

cols = ColumnHelper()


class TestCalcSegStats:
    """Tests for the _calc_seg_stats method."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104, 105],
                "segment_name": ["A", "B", "A", "B", "A"],
                cols.unit_qty: [10, 20, 15, 30, 25],
            },
        )

    def test_correctly_calculates_revenue_transactions_customers_per_segment(self, base_df):
        """Test that the method correctly calculates at the transaction-item level."""
        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [500.0, 500.0, 1000.0],
                cols.agg_transaction_id: [3, 2, 5],
                cols.agg_customer_id: [3, 2, 5],
                cols.agg_unit_qty: [50, 50, 100],
                cols.calc_spend_per_cust: [166.666667, 250.0, 200.0],
                cols.calc_spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
                cols.calc_price_per_unit: [10.0, 10.0, 10.0],
                cols.calc_units_per_trans: [16.666667, 25.0, 20.0],
            },
        )
        segment_stats = (
            SegTransactionStats(base_df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_correctly_calculates_revenue_transactions_customers(self):
        """Test that the method correctly calculates at the transaction level."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104, 105],
                "segment_name": ["A", "B", "A", "B", "A"],
            },
        )

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [500.0, 500.0, 1000.0],
                cols.agg_transaction_id: [3, 2, 5],
                cols.agg_customer_id: [3, 2, 5],
                cols.calc_spend_per_cust: [166.666667, 250.0, 200.0],
                cols.calc_spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
            },
        )

        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_handles_dataframe_with_one_segment(self, base_df):
        """Test that the method correctly handles a DataFrame with only one segment."""
        df = base_df.copy()
        df["segment_name"] = "A"

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "Total"],
                cols.agg_unit_spend: [1000.0, 1000.0],
                cols.agg_transaction_id: [5, 5],
                cols.agg_customer_id: [5, 5],
                cols.agg_unit_qty: [100, 100],
                cols.calc_spend_per_cust: [200.0, 200.0],
                cols.calc_spend_per_trans: [200.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0],
                cols.calc_price_per_unit: [10.0, 10.0],
                cols.calc_units_per_trans: [20.0, 20.0],
            },
        )

        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_handles_dataframe_with_zero_net_units(self, base_df):
        """Test that the method correctly handles a DataFrame with a segment with net zero units."""
        df = base_df.copy()
        df[cols.unit_qty] = [10, 20, 15, 30, -25]

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [500.0, 500.0, 1000.0],
                cols.agg_transaction_id: [3, 2, 5],
                cols.agg_customer_id: [3, 2, 5],
                cols.agg_unit_qty: [0, 50, 50],
                cols.calc_spend_per_cust: [166.666667, 250.0, 200.0],
                cols.calc_spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
                cols.calc_price_per_unit: [np.nan, 10.0, 20.0],
                cols.calc_units_per_trans: [0, 25.0, 10.0],
            },
        )
        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)

        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_excludes_total_row_when_calc_total_false(self, base_df):
        """Test that the method excludes the total row when calc_total=False."""
        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B"],
                cols.agg_unit_spend: [500.0, 500.0],
                cols.agg_transaction_id: [3, 2],
                cols.agg_customer_id: [3, 2],
                cols.agg_unit_qty: [50, 50],
                cols.calc_spend_per_cust: [166.666667, 250.0],
                cols.calc_spend_per_trans: [166.666667, 250.0],
                cols.calc_trans_per_cust: [1.0, 1.0],
                cols.calc_price_per_unit: [10.0, 10.0],
                cols.calc_units_per_trans: [16.666667, 25.0],
            },
        )

        segment_stats = (
            SegTransactionStats(base_df, "segment_name", calc_total=False)
            .df.sort_values("segment_name")
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_calculates_segment_stats_without_customer_data(self, base_df):
        """Test that the method correctly calculates segment statistics without customer data."""
        df_without_customer = base_df.drop(columns=[cols.customer_id])

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [500.0, 500.0, 1000.0],
                cols.agg_transaction_id: [3, 2, 5],
                cols.agg_unit_qty: [50, 50, 100],
                cols.calc_spend_per_trans: [166.666667, 250.0, 200.0],
                cols.calc_price_per_unit: [10.0, 10.0, 10.0],
                cols.calc_units_per_trans: [16.666667, 25.0, 20.0],
            },
        )
        segment_stats = (
            SegTransactionStats(df_without_customer, "segment_name")
            .df.sort_values("segment_name")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(segment_stats, expected_output)


class TestSegTransactionStats:
    """Tests for the SegTransactionStats class."""

    def test_handles_empty_dataframe_with_errors(self):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        df = pd.DataFrame(
            columns=[cols.unit_spend, cols.transaction_id, cols.unit_qty],
        )

        with pytest.raises(ValueError):
            SegTransactionStats(df, "segment_name")

    def test_multiple_segment_columns(self):
        """Test that the class correctly handles multiple segment columns."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "segment_name": [
                    "High Value",
                    "High Value",
                    "Medium Value",
                    "Medium Value",
                    "High Value",
                    "High Value",
                ],
                "region": ["North", "North", "South", "South", "East", "East"],
            },
        )

        # Test with a list of segment columns
        seg_stats = SegTransactionStats(df, ["segment_name", "region"])

        # Create expected DataFrame with the combinations actually produced
        expected_output = pd.DataFrame(
            {
                "segment_name": ["High Value", "High Value", "Medium Value", "Total"],
                "region": ["East", "North", "South", "Total"],
                cols.agg_unit_spend: [650.0, 250.0, 450.0, 1350.0],
                cols.agg_transaction_id: [2, 2, 2, 6],
                cols.agg_customer_id: [1, 1, 1, 3],
                cols.calc_spend_per_cust: [650.0, 250.0, 450.0, 450.0],
                cols.calc_spend_per_trans: [325.0, 125.0, 225.0, 225.0],
                cols.calc_trans_per_cust: [2.0, 2.0, 2.0, 2.0],
            },
        )

        # Sort both dataframes by the segment columns for consistent comparison
        result_df = seg_stats.df.sort_values(["segment_name", "region"]).reset_index(drop=True)
        expected_output = expected_output.sort_values(["segment_name", "region"]).reset_index(drop=True)

        # Check that both segment columns are in the result
        assert "segment_name" in result_df.columns
        assert "region" in result_df.columns

        # Check number of rows - the implementation only returns actual combinations that exist in data
        # plus the Total row, not all possible combinations
        assert len(result_df) == len(expected_output)

        # Use pandas testing to compare the dataframes
        pd.testing.assert_frame_equal(result_df[expected_output.columns], expected_output)

    def test_rollup_disabled(self):
        """Test that rollup rows are not included when calc_rollup is False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with rollup disabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=False,
            calc_total=True,
        )

        result_df = seg_stats.df

        # Test constants
        expected_rows_without_rollup = 5  # 4 detail + 1 grand total

        # Should have:
        # - 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        # - 1 grand total row (Total-Total)
        assert len(result_df) == expected_rows_without_rollup

        # Check for the absence of rollup rows
        rollup_rows = result_df[(result_df["subcategory"] == "Total") & (result_df["category"] != "Total")]
        assert len(rollup_rows) == 0

    def test_custom_rollup_value_string(self):
        """Test using a custom string value for rollup totals."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        custom_value = "ALL"

        # Create SegTransactionStats with a custom rollup value
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            rollup_value=custom_value,
        )

        result_df = seg_stats.df

        # Test constants
        expected_custom_rollup_rows = 3  # 2 category subtotals + 1 grand total

        # Check for the presence of rollup rows with custom value
        rollup_rows = result_df[result_df["subcategory"] == custom_value]
        assert len(rollup_rows) == expected_custom_rollup_rows

        # Verify grand total row uses custom value
        grand_total = result_df[(result_df["category"] == custom_value) & (result_df["subcategory"] == custom_value)]
        assert len(grand_total) == 1

    def test_rollup_with_different_value_types(self):
        """Test rollup with different value types for each column."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
                "product_id": [10, 20, 30, 40, 50, 60],
            },
        )

        # Create SegTransactionStats with a list of different value types
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory", "product_id"],
            calc_rollup=True,
            rollup_value=["ALL", "Subtotal", 0],  # String, String, Integer
        )

        result_df = seg_stats.df

        # Verify that each column uses the correct rollup value type
        assert "category" in result_df.columns
        assert "subcategory" in result_df.columns
        assert "product_id" in result_df.columns

        # Test constants
        expected_level1_rollups = 2  # level-1 rollup rows (category only)

        # Check for level-1 rollup rows (category only)
        level1_rollups = result_df[
            (result_df["subcategory"] == "Subtotal") & (result_df["product_id"] == 0) & (result_df["category"] != "ALL")
        ]
        assert len(level1_rollups) == expected_level1_rollups

        # Verify grand total row uses the specified values
        grand_total = result_df[
            (result_df["category"] == "ALL") & (result_df["subcategory"] == "Subtotal") & (result_df["product_id"] == 0)
        ]
        assert len(grand_total) == 1

    def test_rollup_value_list_wrong_length(self):
        """Test that an error is raised when rollup_value list length doesn't match segment_col length."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "category": ["Clothing", "Footwear", "Electronics"],
                "subcategory": ["Jeans", "Sneakers", "Phones"],
            },
        )

        # Attempt to create SegTransactionStats with mismatched list length
        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(
                df,
                segment_col=["category", "subcategory"],
                calc_rollup=True,
                rollup_value=["Total"],  # Only one value for two columns
            )

        assert "must match the number of segment columns" in str(excinfo.value)

    def test_plot_with_multiple_segment_columns(self):
        """Test that plotting with multiple segment columns raises a ValueError."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "segment_name": ["A", "B", "A"],
                "region": ["North", "South", "East"],
            },
        )

        seg_stats = SegTransactionStats(df, ["segment_name", "region"])

        with pytest.raises(ValueError) as excinfo:
            seg_stats.plot("spend")

        assert "Plotting is only supported for a single segment column" in str(excinfo.value)

    def test_extra_aggs_functionality(self):
        """Test that the extra_aggs parameter works correctly."""
        # Constants for expected values
        segment_a_store_count = 3  # Segment A has stores 1, 2, 4
        segment_b_store_count = 2  # Segment B has stores 1, 3
        total_store_count = 4  # Total has stores 1, 2, 3, 4

        segment_a_product_count = 3  # Segment A has products 10, 20, 40
        segment_b_product_count = 2  # Segment B has products 10, 30
        total_product_count = 4  # Total has products 10, 20, 30, 40
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 3, 3],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "segment_name": ["A", "A", "B", "B", "A", "A"],
                "store_id": [1, 2, 1, 3, 2, 4],
                "product_id": [10, 20, 10, 30, 20, 40],
            },
        )

        # Test with a single extra aggregation
        seg_stats = SegTransactionStats(
            df,
            "segment_name",
            extra_aggs={"distinct_stores": ("store_id", "nunique")},
        )

        # Verify the extra column exists and has correct values
        assert "distinct_stores" in seg_stats.df.columns

        # Sort by segment_name to ensure consistent order
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        assert result_df.loc[0, "distinct_stores"] == segment_a_store_count  # Segment A
        assert result_df.loc[1, "distinct_stores"] == segment_b_store_count  # Segment B
        assert result_df.loc[2, "distinct_stores"] == total_store_count  # Total

        # Test with multiple extra aggregations
        seg_stats_multi = SegTransactionStats(
            df,
            "segment_name",
            extra_aggs={
                "distinct_stores": ("store_id", "nunique"),
                "distinct_products": ("product_id", "nunique"),
            },
        )

        # Verify both extra columns exist
        assert "distinct_stores" in seg_stats_multi.df.columns
        assert "distinct_products" in seg_stats_multi.df.columns

        # Sort by segment_name to ensure consistent order
        result_df_multi = seg_stats_multi.df.sort_values("segment_name").reset_index(drop=True)

        assert result_df_multi["distinct_products"].to_list() == [
            segment_a_product_count,
            segment_b_product_count,
            total_product_count,
        ]

    def test_extra_aggs_with_invalid_column(self):
        """Test that an error is raised when an invalid column is specified in extra_aggs."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "segment_name": ["A", "B", "A"],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, "segment_name", extra_aggs={"invalid_agg": ("nonexistent_column", "nunique")})

        assert "does not exist in the data" in str(excinfo.value)

    def test_extra_aggs_with_invalid_function(self):
        """Test that an error is raised when an invalid function is specified in extra_aggs."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.transaction_id: [101, 102, 103],
                "segment_name": ["A", "B", "A"],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, "segment_name", extra_aggs={"invalid_agg": (cols.customer_id, "invalid_function")})

        assert "not available for column" in str(excinfo.value)

    def test_with_custom_column_names(self):
        """Test SegTransactionStats with custom column names."""
        custom_columns = {
            "column.customer_id": "cust_id",
            "column.unit_spend": "revenue",
            "column.transaction_id": "trans_id",
            "column.unit_quantity": "quantity",
        }

        custom_df = pd.DataFrame(
            {
                "cust_id": [1, 1, 2, 2],
                "revenue": [100.0, 150.0, 200.0, 250.0],
                "trans_id": [101, 102, 103, 104],
                "segment_name": ["A", "A", "B", "B"],
                "quantity": [2, 3, 4, 5],
            },
        )

        with option_context(*[item for pair in custom_columns.items() for item in pair]):
            seg_stats = SegTransactionStats(custom_df, segment_col="segment_name")
            result = seg_stats.df
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

            expected_columns = [cols.agg_customer_id, cols.agg_transaction_id, cols.agg_unit_spend, cols.agg_unit_qty]
            for col in expected_columns:
                assert col in seg_stats.df.columns, f"Expected column {col} missing from output"

    def test_complete_rollup_hierarchy_two_columns(self):
        """Expect prefix and suffix rollups plus grand total when calc_rollup and calc_total are True.

        Expected rows (with rollup_value defaulting to "Total"):
        - Detail: (Clothing, Jeans), (Clothing, Shirts), (Footwear, Jeans), (Footwear, Shirts)
        - Prefix rollups: (Clothing, Total), (Footwear, Total)
        - Suffix rollups: (Total, Jeans), (Total, Shirts)
        - Grand total: (Total, Total)
        Total expected rows = 9.
        """
        # Use a small in-memory dataset (self-contained, no external files)
        df_sample = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0],
                cols.transaction_id: [101, 102, 103, 104],
                "category_0_name": ["Clothing", "Clothing", "Footwear", "Footwear"],
                "category_1_name": ["Jeans", "Shirts", "Jeans", "Shirts"],
            },
        )

        segment_cols = ["category_0_name", "category_1_name"]
        measure_col = cols.unit_spend

        # Run SegTransactionStats with rollups
        seg_stats = SegTransactionStats(
            df_sample,
            segment_col=segment_cols,
            calc_rollup=True,
            calc_total=True,
        )

        result_df = seg_stats.df

        # Convert to dicts for order-insensitive comparison
        records = result_df.to_dict(orient="records")

        # Dynamically compute expected sums
        expected = {}

        # Detail rows
        for (cat0, cat1), group in df_sample.groupby(segment_cols):
            expected[(cat0, cat1)] = group[measure_col].sum()

        # Prefix rollups (category subtotal)
        for cat0, group in df_sample.groupby("category_0_name"):
            expected[(cat0, "Total")] = group[measure_col].sum()

        # Suffix rollups (subcategory subtotal)
        for cat1, group in df_sample.groupby("category_1_name"):
            expected[("Total", cat1)] = group[measure_col].sum()

        # Grand total
        expected[("Total", "Total")] = df_sample[measure_col].sum()

        # Validate each expected row exists and sums match
        for (cat0, cat1), expected_sum in expected.items():
            matches = [r for r in records if r["category_0_name"] == cat0 and r["category_1_name"] == cat1]
            assert len(matches) == 1, f"Missing row for ({cat0}, {cat1})"
            assert matches[0][cols.agg_unit_spend] == expected_sum

    def test_complete_rollup_hierarchy_three_columns(self):
        """Expect prefix + suffix rollups + grand total with 3 segment columns.

        Columns: region, category, subcategory
        Expected rows:
        - Detail: 2 regions x 2 categories x 2 subcategories = 8
        - Prefix rollups:
            (region, category, Total) → 4
            (region, Total, Total) → 2
        - Suffix rollups:
            (Total, category, subcategory) → 4
            (Total, Total, subcategory) → 2
        - Grand total: (Total, Total, Total) → 1

        Total expected rows = 21.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: range(1, 9),
                cols.unit_spend: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                cols.transaction_id: range(101, 109),
                "region": ["North", "North", "North", "North", "South", "South", "South", "South"],
                "category": ["Clothing", "Clothing", "Footwear", "Footwear"] * 2,
                "subcategory": ["Jeans", "Shirts"] * 4,
            },
        )

        seg_stats = SegTransactionStats(
            df,
            segment_col=["region", "category", "subcategory"],
            calc_rollup=True,
            calc_total=True,
        )

        result_df = seg_stats.df

        # Row count check - catches duplicates!
        expected_rows = 21
        assert len(result_df) == expected_rows, f"Expected {expected_rows} rows, got {len(result_df)}"

        # Spot check: one prefix rollup
        north_clothing_total = result_df[
            (result_df["region"] == "North")
            & (result_df["category"] == "Clothing")
            & (result_df["subcategory"] == "Total")
        ]
        assert len(north_clothing_total) == 1
        assert north_clothing_total[cols.agg_unit_spend].values[0] == 10.0 + 20.0

        # Spot check: one suffix rollup (Total, Total, Jeans)
        total_total_jeans = result_df[
            (result_df["region"] == "Total")
            & (result_df["category"] == "Total")
            & (result_df["subcategory"] == "Jeans")
        ]
        assert len(total_total_jeans) == 1
        assert total_total_jeans[cols.agg_unit_spend].values[0] == 10.0 + 30.0 + 50.0 + 70.0

        # Grand total
        grand_total = result_df[
            (result_df["region"] == "Total")
            & (result_df["category"] == "Total")
            & (result_df["subcategory"] == "Total")
        ]
        assert len(grand_total) == 1
        assert grand_total[cols.agg_unit_spend].values[0] == sum([10, 20, 30, 40, 50, 60, 70, 80])

    def test_rollup_enabled_total_disabled(self):
        """Test that rollup rows are included but grand total is excluded when calc_rollup=True, calc_total=False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with rollup enabled but total disabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            calc_total=False,
        )

        result_df = seg_stats.df

        # Expected counts:
        # - 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        # - 2 prefix rollup rows (Clothing-Total, Footwear-Total)
        # - NO suffix rollup rows (to avoid "Total" in category when calc_total=False)
        # - NO grand total row (Total-Total)
        expected_rows_with_rollup_no_total = 6

        assert len(result_df) == expected_rows_with_rollup_no_total

        # Test constants
        expected_prefix_rollups = 2  # Clothing-Total, Footwear-Total

        # Check for presence of prefix rollup rows
        prefix_rollups = result_df[(result_df["subcategory"] == "Total") & (result_df["category"] != "Total")]
        assert len(prefix_rollups) == expected_prefix_rollups

        # Check for absence of suffix rollup rows (should not exist when calc_total=False)
        suffix_rollups = result_df[(result_df["category"] == "Total") & (result_df["subcategory"] != "Total")]
        assert len(suffix_rollups) == 0

        # Check for absence of grand total row
        grand_total = result_df[(result_df["category"] == "Total") & (result_df["subcategory"] == "Total")]
        assert len(grand_total) == 0

    def test_rollup_disabled_total_disabled(self):
        """Test that only detail rows are included when calc_rollup=False, calc_total=False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with both rollup and total disabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=False,
            calc_total=False,
        )

        result_df = seg_stats.df

        # Expected: only 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        expected_rows_detail_only = 4

        assert len(result_df) == expected_rows_detail_only

        # Check for absence of any rollup rows
        rollup_rows = result_df[(result_df["subcategory"] == "Total") | (result_df["category"] == "Total")]
        assert len(rollup_rows) == 0

        # Verify all rows are detail rows (no "Total" values)
        assert "Total" not in result_df["category"].values
        assert "Total" not in result_df["subcategory"].values


class TestUnknownCustomerTracking:
    """Tests for unknown customer tracking functionality."""

    @pytest.mark.parametrize(
        ("unknown_value", "customer_ids"),
        [
            (-1, [1, 2, -1, 3]),  # int value
            ("UNKNOWN", ["C1", "C2", "UNKNOWN", "C3"]),  # string value
        ],
    )
    def test_unknown_customer_input_types(self, unknown_value, customer_ids):
        """Test unknown customer tracking with different input value types."""
        df = pd.DataFrame(
            {
                cols.customer_id: customer_ids,
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104],
                "segment_name": ["A", "A", "A", "B"],
            },
        )

        seg_stats = SegTransactionStats(df, "segment_name", unknown_customer_value=unknown_value)
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [300.0, 300.0, 600.0],
                cols.agg_transaction_id: [2, 1, 3],
                cols.agg_customer_id: [2, 1, 3],
                cols.calc_spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc_spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
                cols.agg_unit_spend_unknown: [150.0, 0.0, 150.0],
                cols.agg_transaction_id_unknown: [1, 0, 1],
                cols.calc_spend_per_trans_unknown: [150.0, np.nan, 150.0],
                cols.agg_unit_spend_total: [450.0, 300.0, 750.0],
                cols.agg_transaction_id_total: [3, 1, 4],
                cols.calc_spend_per_trans_total: [150.0, 300.0, 187.5],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_with_ibis_literal(self):
        """Test unknown customer tracking with ibis literal."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, -1, 3],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104],
                "segment_name": ["A", "A", "A", "B"],
            },
        )

        seg_stats = SegTransactionStats(df, "segment_name", unknown_customer_value=ibis.literal(-1))
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [300.0, 300.0, 600.0],
                cols.agg_transaction_id: [2, 1, 3],
                cols.agg_customer_id: [2, 1, 3],
                cols.calc_spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc_spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
                cols.agg_unit_spend_unknown: [150.0, 0.0, 150.0],
                cols.agg_transaction_id_unknown: [1, 0, 1],
                cols.calc_spend_per_trans_unknown: [150.0, np.nan, 150.0],
                cols.agg_unit_spend_total: [450.0, 300.0, 750.0],
                cols.agg_transaction_id_total: [3, 1, 4],
                cols.calc_spend_per_trans_total: [150.0, 300.0, 187.5],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_with_boolean_expression(self):
        """Test unknown customer tracking with boolean expression."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, -1, -2, 3],
                cols.unit_spend: [100.0, 200.0, 150.0, 250.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104, 105],
                "segment_name": ["A", "A", "A", "A", "B"],
            },
        )

        data_table = ibis.memtable(df)
        seg_stats = SegTransactionStats(
            data_table,
            "segment_name",
            unknown_customer_value=data_table[cols.customer_id] < 0,
        )
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [300.0, 300.0, 600.0],
                cols.agg_transaction_id: [2, 1, 3],
                cols.agg_customer_id: [2, 1, 3],
                cols.calc_spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc_spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
                cols.agg_unit_spend_unknown: [400.0, 0.0, 400.0],
                cols.agg_transaction_id_unknown: [2, 0, 2],
                cols.calc_spend_per_trans_unknown: [200.0, np.nan, 200.0],
                cols.agg_unit_spend_total: [700.0, 300.0, 1000.0],
                cols.agg_transaction_id_total: [4, 1, 5],
                cols.calc_spend_per_trans_total: [175.0, 300.0, 200.0],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_with_quantity(self):
        """Test unknown customer tracking with quantity columns."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, -1, 3],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0],
                cols.transaction_id: [101, 102, 103, 104],
                cols.unit_qty: [10, 20, 15, 30],
                "segment_name": ["A", "A", "A", "B"],
            },
        )

        seg_stats = SegTransactionStats(df, "segment_name", unknown_customer_value=-1)
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                cols.agg_unit_spend: [300.0, 300.0, 600.0],
                cols.agg_transaction_id: [2, 1, 3],
                cols.agg_customer_id: [2, 1, 3],
                cols.agg_unit_qty: [30, 30, 60],
                cols.calc_spend_per_cust: [150.0, 300.0, 200.0],
                cols.calc_spend_per_trans: [150.0, 300.0, 200.0],
                cols.calc_trans_per_cust: [1.0, 1.0, 1.0],
                cols.calc_price_per_unit: [10.0, 10.0, 10.0],
                cols.calc_units_per_trans: [15.0, 30.0, 20.0],
                cols.agg_unit_spend_unknown: [150.0, 0.0, 150.0],
                cols.agg_transaction_id_unknown: [1, 0, 1],
                cols.agg_unit_qty_unknown: [15, 0, 15],
                cols.calc_spend_per_trans_unknown: [150.0, np.nan, 150.0],
                cols.calc_price_per_unit_unknown: [10.0, np.nan, 10.0],
                cols.calc_units_per_trans_unknown: [15.0, np.nan, 15.0],
                cols.agg_unit_spend_total: [450.0, 300.0, 750.0],
                cols.agg_transaction_id_total: [3, 1, 4],
                cols.agg_unit_qty_total: [45, 30, 75],
                cols.calc_spend_per_trans_total: [150.0, 300.0, 187.5],
                cols.calc_price_per_unit_total: [10.0, 10.0, 10.0],
                cols.calc_units_per_trans_total: [15.0, 30.0, 18.75],
            },
        )

        pd.testing.assert_frame_equal(result_df, expected_output)

    def test_unknown_customer_error_when_customer_id_missing(self):
        """Test that error is raised when customer_id column is missing."""
        df = pd.DataFrame(
            {
                cols.unit_spend: [100.0, 200.0],
                cols.transaction_id: [101, 102],
                "segment_name": ["A", "B"],
            },
        )

        with pytest.raises(ValueError) as excinfo:
            SegTransactionStats(df, "segment_name", unknown_customer_value=-1)

        assert "required when unknown_customer_value parameter is specified" in str(excinfo.value)

    def test_unknown_customer_with_rollups(self):
        """Test unknown customer tracking with rollups enabled."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, -1, 2, -1],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                "category": ["Clothing", "Clothing", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Shirts", "Sneakers", "Boots"],
            },
        )

        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            unknown_customer_value=-1,
        )
        result_df = seg_stats.df

        # Check that rollup rows include unknown and total columns
        total_row = result_df[(result_df["category"] == "Total") & (result_df["subcategory"] == "Total")]
        assert len(total_row) == 1
        assert cols.agg_unit_spend_unknown in total_row.columns
        assert cols.agg_unit_spend_total in total_row.columns
        expected_unknown_spend = 400.0
        expected_total_spend = 700.0
        assert total_row[cols.agg_unit_spend_unknown].iloc[0] == expected_unknown_spend
        assert total_row[cols.agg_unit_spend_total].iloc[0] == expected_total_spend

    def test_unknown_customer_with_extra_aggs(self):
        """Test unknown customer tracking with extra aggregations."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, -1, 2, -1],
                cols.unit_spend: [100.0, 150.0, 200.0, 250.0],
                cols.transaction_id: [101, 102, 103, 104],
                "segment_name": ["A", "A", "B", "B"],
                "store_id": [1, 2, 1, 3],
            },
        )

        seg_stats = SegTransactionStats(
            df,
            "segment_name",
            unknown_customer_value=-1,
            extra_aggs={"stores": ("store_id", "nunique")},
        )
        result_df = seg_stats.df.sort_values("segment_name").reset_index(drop=True)

        # Check that extra agg has three variants
        assert "stores" in result_df.columns
        assert "stores_unknown" in result_df.columns
        assert "stores_total" in result_df.columns

        # Verify values for segment A
        expected_identified_stores = 1  # Segment A identified: store 1
        expected_unknown_stores = 1  # Segment A unknown: store 2
        expected_total_stores = 2  # Segment A total: stores 1, 2
        assert result_df.loc[0, "stores"] == expected_identified_stores
        assert result_df.loc[0, "stores_unknown"] == expected_unknown_stores
        assert result_df.loc[0, "stores_total"] == expected_total_stores
