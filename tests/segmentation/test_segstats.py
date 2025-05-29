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

    @pytest.mark.parametrize(
        ("segment_config", "calc_total", "calc_rollup", "extra_aggs_config"),
        [
            # Single segment column
            ("single", True, False, None),
            ("single", False, False, None),
            # Multiple segment columns
            ("multiple", True, False, None),
            ("multiple", True, True, None),
            # With extra aggregations
            ("single", True, False, "simple"),
            ("multiple", True, True, "complex"),
        ],
    )
    def test_with_custom_column_names(self, segment_config, calc_total, calc_rollup, extra_aggs_config):
        """Test that SegTransactionStats works correctly with completely renamed columns."""
        custom_columns = {
            "column.customer_id": "custom_cust_id",
            "column.unit_spend": "custom_revenue",
            "column.transaction_id": "custom_trans_id",
            "column.unit_quantity": "custom_quantity",
        }

        base_data, segment_col = self._create_test_data(segment_config)
        custom_df = self._apply_custom_column_names(base_data, custom_columns)
        extra_aggs = self._get_extra_aggs_config(extra_aggs_config)
        rollup_value = self._get_rollup_value(segment_col)

        with option_context(*[item for pair in custom_columns.items() for item in pair]):
            seg_stats = SegTransactionStats(
                custom_df,
                segment_col=segment_col,
                calc_total=calc_total,
                calc_rollup=calc_rollup,
                rollup_value=rollup_value,
                extra_aggs=extra_aggs,
            )

            self._validate_basic_structure(seg_stats, segment_col)
            self._validate_core_columns(seg_stats.df)
            self._validate_extra_aggregations(seg_stats.df, extra_aggs)
            self._validate_total_rows(seg_stats.df, segment_col, calc_total, rollup_value)
            self._validate_rollup_rows(seg_stats.df, segment_col, calc_rollup, rollup_value)
            self._validate_numeric_data(seg_stats.df)
            self._validate_plotting(seg_stats, segment_col)
            self._validate_calculated_metrics(seg_stats.df)

    def _create_test_data(self, segment_config):
        """Create base test data and determine segment column configuration."""
        base_data = {
            get_option("column.customer_id"): [1, 1, 2, 2, 3, 3, 4, 4],
            cols.unit_spend: [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0],
            cols.transaction_id: [101, 102, 103, 104, 105, 106, 107, 108],
            cols.unit_qty: [10, 15, 20, 25, 30, 35, 40, 45],
            "store_id": [1, 2, 1, 3, 2, 4, 1, 2],
            "product_id": [10, 20, 10, 30, 20, 40, 50, 60],
        }

        if segment_config == "single":
            base_data["segment_name"] = [
                "Premium",
                "Premium",
                "Standard",
                "Standard",
                "Premium",
                "Premium",
                "Standard",
                "Standard",
            ]
            segment_col = "segment_name"
        else:
            base_data["category"] = [
                "Electronics",
                "Electronics",
                "Clothing",
                "Clothing",
                "Electronics",
                "Electronics",
                "Footwear",
                "Footwear",
            ]
            base_data["region"] = ["North", "South", "North", "South", "East", "West", "North", "South"]
            segment_col = ["category", "region"]

        return base_data, segment_col

    def _apply_custom_column_names(self, base_data, custom_columns):
        """Apply custom column names to the base data."""
        df = pd.DataFrame(base_data)
        rename_mapping = {
            get_option("column.customer_id"): custom_columns["column.customer_id"],
            cols.unit_spend: custom_columns["column.unit_spend"],
            cols.transaction_id: custom_columns["column.transaction_id"],
            cols.unit_qty: custom_columns["column.unit_quantity"],
        }
        return df.rename(columns=rename_mapping)

    def _get_extra_aggs_config(self, extra_aggs_config):
        """Get extra aggregations configuration based on test parameter."""
        if extra_aggs_config == "simple":
            return {"distinct_stores": ("store_id", "nunique")}
        if extra_aggs_config == "complex":
            return {
                "distinct_stores": ("store_id", "nunique"),
                "distinct_products": ("product_id", "nunique"),
            }
        return None

    def _get_rollup_value(self, segment_col):
        """Get appropriate rollup value based on segment column configuration."""
        return "Total" if not isinstance(segment_col, list) else ["All_Categories", "All_Regions"]

    def _validate_basic_structure(self, seg_stats, segment_col):
        """Validate basic structure of SegTransactionStats result."""
        assert seg_stats is not None
        result_df = seg_stats.df
        assert not result_df.empty

        if isinstance(segment_col, str):
            assert segment_col in result_df.columns
        else:
            for col in segment_col:
                assert col in result_df.columns

    def _validate_core_columns(self, result_df):
        """Validate that all expected core columns are present."""
        expected_core_cols = [
            cols.agg_unit_spend,
            cols.agg_transaction_id,
            cols.agg_customer_id,
            cols.agg_unit_qty,
            cols.calc_spend_per_cust,
            cols.calc_spend_per_trans,
            cols.calc_trans_per_cust,
            cols.calc_price_per_unit,
            cols.calc_units_per_trans,
        ]

        for col in expected_core_cols:
            assert col in result_df.columns, f"Expected column {col} not found in result"

    def _validate_extra_aggregations(self, result_df, extra_aggs):
        """Validate extra aggregation columns if specified."""
        if extra_aggs:
            for agg_name in extra_aggs:
                assert agg_name in result_df.columns, f"Expected extra agg column {agg_name} not found"

    def _validate_total_rows(self, result_df, segment_col, calc_total, rollup_value):
        """Validate total row presence based on calc_total parameter."""
        if not calc_total:
            return

        if isinstance(segment_col, str):
            total_rows = result_df[result_df[segment_col] == "Total"]
        else:
            # For multiple columns, check if total row exists with rollup values
            total_condition = True
            for i, col in enumerate(segment_col):
                expected_val = rollup_value[i] if isinstance(rollup_value, list) else rollup_value
                total_condition &= result_df[col] == expected_val
            total_rows = result_df[total_condition]

        assert len(total_rows) >= 1, "Total row should be present when calc_total=True"

    def _validate_rollup_rows(self, result_df, segment_col, calc_rollup, rollup_value):
        """Validate rollup rows if enabled (only for multiple segment columns)."""
        if not (calc_rollup and isinstance(segment_col, list)):
            return

        rollup_condition = False
        for i, col in enumerate(segment_col[1:], 1):
            expected_val = rollup_value[i] if isinstance(rollup_value, list) else rollup_value
            rollup_condition |= result_df[col] == expected_val

        rollup_rows = result_df[rollup_condition]
        assert len(rollup_rows) > 0, "Rollup rows should be present when calc_rollup=True"

    def _validate_numeric_data(self, result_df):
        """Validate numeric data integrity."""
        numeric_cols = [cols.agg_unit_spend, cols.agg_transaction_id, cols.agg_customer_id]
        for col in numeric_cols:
            assert result_df[col].notna().all(), f"Column {col} should not have NaN values"
            assert (result_df[col] >= 0).all(), f"Column {col} should have non-negative values"

    def _validate_plotting(self, seg_stats, segment_col):
        """Test plotting functionality for single segment column."""
        if not isinstance(segment_col, str):
            return

        try:
            ax = seg_stats.plot(cols.agg_unit_spend, hide_total=True)
            assert ax is not None, "Plot should be created successfully"
        except (ValueError, TypeError) as e:
            pytest.fail(f"Plotting failed with custom columns: {e}")

    def _validate_calculated_metrics(self, result_df):
        """Validate calculated metrics have expected properties."""
        assert (result_df[cols.calc_spend_per_cust] > 0).all(), "Spend per customer should be positive"
        assert (result_df[cols.calc_spend_per_trans] > 0).all(), "Spend per transaction should be positive"
        assert (result_df[cols.calc_trans_per_cust] > 0).all(), "Transactions per customer should be positive"
