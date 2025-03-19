"""Tests for the SegTransactionStats class."""

import ibis
import numpy as np
import pandas as pd
import pytest

from pyretailscience.analysis.segmentation import (
    HMLSegmentation,
    RFMSegmentation,
    SegTransactionStats,
    ThresholdSegmentation,
)
from pyretailscience.options import ColumnHelper, get_option

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
                cols.customers_pct: [0.6, 0.4, 1.0],
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
                cols.customers_pct: [0.6, 0.4, 1.0],
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
                cols.customers_pct: [1.0, 1.0],
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
                cols.customers_pct: [0.6, 0.4, 1.0],
            },
        )
        segment_stats = SegTransactionStats(df, "segment_name").df.sort_values("segment_name").reset_index(drop=True)

        pd.testing.assert_frame_equal(segment_stats, expected_output)


class TestThresholdSegmentation:
    """Tests for the ThresholdSegmentation class."""

    def test_correct_segmentation(self):
        """Test that the method correctly segments customers based on given thresholds and segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4],
                cols.unit_spend: [100, 200, 300, 400],
            },
        )
        thresholds = [0.5, 1]
        segments = ["Low", "High"]
        seg = ThresholdSegmentation(
            df=df,
            thresholds=thresholds,
            segments=segments,
            value_col=cols.unit_spend,
            zero_value_customers="exclude",
        )
        result_df = seg.df
        assert result_df.loc[1, "segment_name"] == "Low"
        assert result_df.loc[2, "segment_name"] == "Low"
        assert result_df.loc[3, "segment_name"] == "High"
        assert result_df.loc[4, "segment_name"] == "High"

    def test_single_customer(self):
        """Test that the method correctly segments a DataFrame with only one customer."""
        df = pd.DataFrame({get_option("column.customer_id"): [1], cols.unit_spend: [100]})
        thresholds = [0.5, 1]
        segments = ["Low"]
        with pytest.raises(ValueError):
            ThresholdSegmentation(
                df=df,
                thresholds=thresholds,
                segments=segments,
            )

    def test_correct_aggregation_function(self):
        """Test that the correct aggregation function is applied for product_id custom segmentation."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
                "product_id": [3, 4, 4, 6, 1, 5, 7, 2, 2, 3, 2, 3, 4, 1],
            },
        )
        value_col = "product_id"
        agg_func = "nunique"

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=[0.2, 0.8, 1],
            segments=["Low", "Medium", "High"],
            zero_value_customers="separate_segment",
        )

        expected_result = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                "product_id": [1, 4, 2, 2, 3],
                "segment_name": ["Low", "High", "Medium", "Medium", "Medium"],
            },
        )
        pd.testing.assert_frame_equal(my_seg.df.sort_values(cols.customer_id).reset_index(), expected_result)

    def test_correctly_checks_segment_data(self):
        """Test that the method correctly merges segment data back into the original DataFrame."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 0, 150, 0],
            },
        )
        value_col = cols.unit_spend
        agg_func = "sum"
        thresholds = [0.33, 0.66, 1]
        segments = ["Low", "Medium", "High"]
        zero_value_customers = "separate_segment"

        # Create ThresholdSegmentation instance
        threshold_seg = ThresholdSegmentation(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
        )

        # Call add_segment method
        segmented_df = threshold_seg.add_segment(df)

        # Assert the correct segment_name
        expected_df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 0, 150, 0],
                "segment_name": ["Low", "High", "Zero", "Medium", "Zero"],
            },
        )
        pd.testing.assert_frame_equal(segmented_df, expected_df)

    def test_handles_dataframe_with_duplicate_customer_id_entries(self):
        """Test that the method correctly handles a DataFrame with duplicate customer_id entries."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 1, 2, 3],
                cols.unit_spend: [100, 200, 300, 150, 250, 350],
            },
        )

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=cols.unit_spend,
            agg_func="sum",
            thresholds=[0.5, 0.8, 1],
            segments=["Light", "Medium", "Heavy"],
            zero_value_customers="include_with_light",
        )

        result_df = my_seg.add_segment(df)
        assert len(result_df) == len(df)

    def test_thresholds_not_unique(self):
        """Test that the method raises an error when the thresholds are not unique."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.5, 0.5, 0.8, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_few_segments(self):
        """Test that the method raises an error when there are too few/many segments for the number of thresholds."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 0.6, 0.8, 1]
        segments = ["Low", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_too_few_thresholds(self):
        """Test that the method raises an error when there are too few/many thresholds for the number of segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        thresholds = [0.2, 0.5, 0.6, 0.8, 1]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)


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
                "segment_name": ["A", "A", "B", "B", "A", "A"],
                "region": ["North", "North", "South", "South", "East", "East"],
            },
        )

        # Test with a list of segment columns
        seg_stats = SegTransactionStats(df, ["segment_name", "region"])

        # Create expected DataFrame with the combinations actually produced
        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "A", "B", "Total"],
                "region": ["East", "North", "South", "Total"],
                cols.agg_unit_spend: [650.0, 250.0, 450.0, 1350.0],
                cols.agg_transaction_id: [2, 2, 2, 6],
                cols.agg_customer_id: [1, 1, 1, 3],
                cols.calc_spend_per_cust: [650.0, 250.0, 450.0, 450.0],
                cols.calc_spend_per_trans: [325.0, 125.0, 225.0, 225.0],
                cols.calc_trans_per_cust: [2.0, 2.0, 2.0, 2.0],
                cols.customers_pct: [1 / 3, 1 / 3, 1 / 3, 1.0],
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
        seg_stats = SegTransactionStats(df, "segment_name", extra_aggs={"distinct_stores": ("store_id", "nunique")})

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


class TestHMLSegmentation:
    """Tests for the HMLSegmentation class."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [1000, 200, 0, 500, 300],
            },
        )

    # Correctly handles zero spend customers when zero_value_customers is "exclude"
    def test_handles_zero_spend_customers_are_excluded_in_result(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "exclude"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="exclude")
        result_df = hml_segmentation.df

        zero_spend_customer_id = 3

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert zero_spend_customer_id not in result_df.index
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"

    # Correctly handles zero spend customers when zero_value_customers is "include_with_light"
    def test_handles_zero_spend_customers_include_with_light(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "include_with_light"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="include_with_light")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[3, "segment_name"] == "Light"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"

    # Correctly handles zero spend customers when zero_value_customers is "separate_segment"
    def test_handles_zero_spend_customers_separate_segment(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "separate_segment"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="separate_segment")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[3, "segment_name"] == "Zero"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"

    # Raises ValueError if required columns are missing
    def test_raises_value_error_if_required_columns_missing(self, base_df):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        with pytest.raises(ValueError):
            HMLSegmentation(base_df.drop(columns=[get_option("column.customer_id")]))

    # Validate that the input dataframe is not changed
    def test_input_dataframe_not_changed(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()

        hml_segmentation = HMLSegmentation(base_df)
        _ = hml_segmentation.df

        assert original_df.equals(base_df)  # Check if the original dataframe is not changed

    def test_alternate_value_col(self, base_df):
        """Test that the method correctly segments a DataFrame with an alternate value column."""
        base_df = base_df.rename(columns={cols.unit_spend: cols.unit_qty})
        hml_segmentation = HMLSegmentation(base_df, value_col=cols.unit_qty)
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"


class TestRFMSegmentation:
    """Tests for the RFMSegmentation class."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                cols.transaction_id: [101, 102, 103, 104, 105],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_date: [
                    "2025-03-01",
                    "2025-02-15",
                    "2025-01-30",
                    "2025-03-10",
                    "2025-02-20",
                ],
            },
        )

    @pytest.fixture
    def expected_df(self):
        """Returns the expected DataFrame for testing segmentation."""
        return pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "frequency": [1, 1, 1, 1, 1],
                "monetary": [100.0, 200.0, 150.0, 300.0, 250.0],
                "r_score": [1, 3, 4, 0, 2],
                "f_score": [0, 1, 2, 3, 4],
                "m_score": [0, 2, 1, 4, 3],
                "rfm_segment": [100, 312, 421, 34, 243],
                "fm_segment": [0, 12, 21, 34, 43],
            },
        ).set_index("customer_id")

    def test_correct_rfm_segmentation(self, base_df, expected_df):
        """Test that the RFM segmentation correctly calculates the RFM scores and segments."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)
        result_df = rfm_segmentation.df
        expected_df["recency_days"] = [16, 30, 46, 7, 25]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_df["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    def test_handles_dataframe_with_missing_columns(self):
        """Test that the method raises an error when required columns are missing."""
        base_df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 150.0],
                cols.transaction_id: [101, 102, 103],
            },
        )

        with pytest.raises(ValueError):
            RFMSegmentation(df=base_df, current_date="2025-03-17")

    def test_single_customer(self):
        """Test that the method correctly calculates RFM segmentation for a single customer."""
        df_single_customer = pd.DataFrame(
            {
                cols.customer_id: [1],
                cols.transaction_id: [101],
                cols.unit_spend: [200.0],
                cols.transaction_date: ["2025-03-01"],
            },
        )
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=df_single_customer, current_date=current_date)
        result_df = rfm_segmentation.df
        assert result_df.loc[1, "rfm_segment"] == 0

    def test_multiple_transactions_per_customer(self):
        """Test that the method correctly handles multiple transactions for the same customer."""
        df_multiple_transactions = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 1, 1],
                cols.transaction_id: [101, 102, 103, 104, 105],
                cols.unit_spend: [120.0, 250.0, 180.0, 300.0, 220.0],
                cols.transaction_date: [
                    "2025-03-01",
                    "2025-02-15",
                    "2025-01-10",
                    "2025-03-10",
                    "2025-02-25",
                ],
            },
        )
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=df_multiple_transactions, current_date=current_date)
        result_df = rfm_segmentation.df

        assert result_df.loc[1, "rfm_segment"] == 0

    def test_calculates_rfm_correctly_for_all_customers(self, base_df):
        """Test that RFM scores are calculated correctly for all customers."""
        current_date = "2025-03-17"
        expected_customer_count = 5
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)
        result_df = rfm_segmentation.df

        assert len(result_df) == expected_customer_count
        assert "rfm_segment" in result_df.columns

    def test_rfm_segmentation_with_no_date(self, base_df, expected_df):
        """Test that the RFM segmentation correctly calculates the RFM scores and segments."""
        rfm_segmentation = RFMSegmentation(df=base_df)
        result_df = rfm_segmentation.df
        expected_df["recency_days"] = [18, 32, 48, 9, 27]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_df["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    def test_invalid_current_date_type(self, base_df):
        """Test that RFMSegmentation raises a TypeError when an invalid current_date is provided."""
        with pytest.raises(
            TypeError,
            match="current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None",
        ):
            RFMSegmentation(base_df, current_date=12345)

    def test_invalid_df_type(self):
        """Test that RFMSegmentation raises a TypeError when df is neither a DataFrame nor an Ibis Table."""
        invalid_df = "this is not a dataframe"

        with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an Ibis Table"):
            RFMSegmentation(df=invalid_df, current_date="2025-03-17")

    def test_ibis_table_property(self, base_df):
        """Test that ibis_table property returns an Ibis Table."""
        segmentation = RFMSegmentation(df=base_df, current_date="2025-03-17")

        result = segmentation.ibis_table

        assert isinstance(result, ibis.Table), "Expected ibis.Table but got a different type"
