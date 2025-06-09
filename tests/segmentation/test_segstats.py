"""Tests for the SegTransactionStats class."""

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

    def test_rollup_with_two_segment_columns(self):
        """Test rollup functionality with two segment columns."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
            },
        )

        # Create SegTransactionStats with rollup enabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory"],
            calc_rollup=True,
            rollup_value="Subtotal",
        )

        result_df = seg_stats.df

        # Verify the structure of the result
        assert "category" in result_df.columns
        assert "subcategory" in result_df.columns

        # Test constants
        expected_total_rows = 7  # 4 detail + 2 rollup + 1 grand total
        expected_rollup_rows = 3  # 2 category subtotals + 1 grand total
        expected_clothing_spend = 600.0  # Sum of Clothing items
        expected_footwear_spend = 1500.0  # Sum of Footwear items
        expected_total_spend = 2100.0  # Sum of all values

        # Should have:
        # - 4 detail rows (Clothing-Jeans, Clothing-Shirts, Footwear-Sneakers, Footwear-Boots)
        # - 2 rollup rows (Clothing-Subtotal, Footwear-Subtotal)
        # - 1 grand total row (Subtotal-Subtotal)
        assert len(result_df) == expected_total_rows

        # Check for the presence of rollup rows
        rollup_rows = result_df[result_df["subcategory"] == "Subtotal"]
        assert len(rollup_rows) == expected_rollup_rows

        # Verify Clothing category rollup
        clothing_rollup = result_df[(result_df["category"] == "Clothing") & (result_df["subcategory"] == "Subtotal")]
        assert len(clothing_rollup) == 1
        assert clothing_rollup[cols.agg_unit_spend].values[0] == expected_clothing_spend

        # Verify Footwear category rollup
        footwear_rollup = result_df[(result_df["category"] == "Footwear") & (result_df["subcategory"] == "Subtotal")]
        assert len(footwear_rollup) == 1
        assert footwear_rollup[cols.agg_unit_spend].values[0] == expected_footwear_spend

        # Verify grand total row
        grand_total = result_df[(result_df["category"] == "Subtotal") & (result_df["subcategory"] == "Subtotal")]
        assert len(grand_total) == 1
        assert grand_total[cols.agg_unit_spend].values[0] == expected_total_spend

    def test_rollup_with_three_segment_columns(self):
        """Test rollup functionality with three segment columns."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_id: [101, 102, 103, 104, 105, 106],
                "category": ["Clothing", "Clothing", "Clothing", "Footwear", "Footwear", "Footwear"],
                "subcategory": ["Jeans", "Jeans", "Shirts", "Sneakers", "Boots", "Boots"],
                "brand": ["Levi's", "Wrangler", "Ralph Lauren", "Nike", "Timberland", "Dr. Martens"],
            },
        )

        # Create SegTransactionStats with rollup enabled
        seg_stats = SegTransactionStats(
            df,
            segment_col=["category", "subcategory", "brand"],
            calc_rollup=True,
        )

        result_df = seg_stats.df

        # Verify the structure of the result
        assert "category" in result_df.columns
        assert "subcategory" in result_df.columns
        assert "brand" in result_df.columns

        # Test constants
        expected_total_rows = 13  # 6 detail + 4 level-2 + 2 level-1 + 1 grand total
        expected_level2_rollups = 4  # category+subcategory combinations with brand=Total
        expected_level1_rollups = 2  # category only, with subcategory=Total
        expected_total_spend = 2100.0  # Sum of all values

        # Expected rows:
        # - 6 detail rows (various combinations)
        # - 4 level-2 rollup rows (category+subcategory combinations with brand=Total)
        # - 2 level-1 rollup rows (category only, with subcategory=Total)
        # - 1 grand total row (all Total)
        assert len(result_df) == expected_total_rows

        # Check for the presence of level-2 rollup rows (category+subcategory)
        level2_rollups = result_df[
            (result_df["brand"] == "Total") & (result_df["subcategory"] != "Total") & (result_df["category"] != "Total")
        ]
        assert len(level2_rollups) == expected_level2_rollups

        # Check for the presence of level-1 rollup rows (category only)
        level1_rollups = result_df[
            (result_df["brand"] == "Total") & (result_df["subcategory"] == "Total") & (result_df["category"] != "Total")
        ]
        assert len(level1_rollups) == expected_level1_rollups

        # Verify grand total row
        grand_total = result_df[
            (result_df["category"] == "Total") & (result_df["subcategory"] == "Total") & (result_df["brand"] == "Total")
        ]
        assert len(grand_total) == 1
        assert grand_total[cols.agg_unit_spend].values[0] == expected_total_spend

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
