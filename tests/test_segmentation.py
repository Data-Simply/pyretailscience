"""Tests for the SegTransactionStats class."""

import pandas as pd
import pytest

from pyretailscience.segmentation import SegTransactionStats


class TestCalcSegStats:
    """Tests for the _calc_seg_stats method."""

    @pytest.fixture()
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "total_price": [100, 200, 150, 300, 250],
                "transaction_id": [101, 102, 103, 104, 105],
                "segment_id": ["A", "B", "A", "B", "A"],
                "quantity": [10, 20, 15, 30, 25],
            },
        )

    def test_correctly_calculates_revenue_transactions_customers_per_segment(self, base_df):
        """Test that the method correctly calculates at the transaction-item level."""
        expected_output = pd.DataFrame(
            {
                "revenue": [500, 500, 1000],
                "transactions": [3, 2, 5],
                "customers": [3, 2, 5],
                "total_quantity": [50, 50, 100],
                "price_per_unit": [10.0, 10.0, 10.0],
                "quantity_per_transaction": [16.666667, 25.0, 20.0],
            },
            index=["A", "B", "total"],
        )

        segment_stats = SegTransactionStats._calc_seg_stats(base_df, "segment_id")
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_correctly_calculates_revenue_transactions_customers(self):
        """Test that the method correctly calculates at the transaction level."""
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "total_price": [100, 200, 150, 300, 250],
                "transaction_id": [101, 102, 103, 104, 105],
                "segment_id": ["A", "B", "A", "B", "A"],
            },
        )

        expected_output = pd.DataFrame(
            {
                "revenue": [500, 500, 1000],
                "transactions": [3, 2, 5],
                "customers": [3, 2, 5],
            },
            index=["A", "B", "total"],
        )

        segment_stats = SegTransactionStats._calc_seg_stats(df, "segment_id")
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_does_not_alter_original_dataframe(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()
        _ = SegTransactionStats._calc_seg_stats(base_df, "segment_id")

        pd.testing.assert_frame_equal(base_df, original_df)

    def test_handles_dataframe_with_one_segment(self, base_df):
        """Test that the method correctly handles a DataFrame with only one segment."""
        df = base_df.copy()
        df["segment_id"] = "A"

        expected_output = pd.DataFrame(
            {
                "revenue": [1000, 1000],
                "transactions": [5, 5],
                "customers": [5, 5],
                "total_quantity": [100, 100],
                "price_per_unit": [10.0, 10.0],
                "quantity_per_transaction": [20.0, 20.0],
            },
            index=["A", "total"],
        )

        segment_stats = SegTransactionStats._calc_seg_stats(df, "segment_id")
        pd.testing.assert_frame_equal(segment_stats, expected_output)


class TestSegTransactionStats:
    """Tests for the SegTransactionStats class."""

    def test_handles_empty_dataframe_with_errors(self):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        df = pd.DataFrame(columns=["total_price", "transaction_id", "segment_id", "quantity"])

        with pytest.raises(ValueError):
            SegTransactionStats(df, "segment_id")
