"""Tests for the SegTransactionStats class."""

import pandas as pd
import pytest

from pyretailscience.segmentation import HMLSegmentation, SegTransactionStats


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


class TestHMLSegmentation:
    """Tests for the HMLSegmentation class."""

    @pytest.fixture()
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "total_price": [1000, 200, 0, 500, 300]})

    def test_no_transactions(self):
        """Test that the method raises an error when there are no transactions."""
        data = {"customer_id": [], "total_price": []}
        df = pd.DataFrame(data)
        with pytest.raises(ValueError):
            HMLSegmentation(df)

    # Correctly handles zero spend customers when zero_value_customers is "exclude"
    def test_handles_zero_spend_customers_are_excluded_in_result(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "exclude"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="exclude")
        result_df = hml_segmentation.df

        zero_spend_customer_id = 3

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert zero_spend_customer_id not in result_df.index
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"

    # Correctly handles zero spend customers when zero_value_customers is "include_with_light"
    def test_handles_zero_spend_customers_include_with_light(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "include_with_light"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="include_with_light")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert result_df.loc[3, "segment_name"] == "Light"
        assert result_df.loc[3, "segment_id"] == "L"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"

    # Correctly handles zero spend customers when zero_value_customers is "separate_segment"
    def test_handles_zero_spend_customers_separate_segment(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "separate_segment"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="separate_segment")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert result_df.loc[3, "segment_name"] == "Zero"
        assert result_df.loc[3, "segment_id"] == "Z"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"

    # Raises ValueError if required columns are missing
    def test_raises_value_error_if_required_columns_missing(self, base_df):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        with pytest.raises(ValueError):
            HMLSegmentation(base_df.drop(columns=["customer_id"]))

    # DataFrame with only one customer
    def test_segments_customer_single(self):
        """Test that the method correctly segments a DataFrame with only one customer."""
        data = {"customer_id": [1], "total_price": [0]}
        df = pd.DataFrame(data)
        with pytest.raises(ValueError):
            HMLSegmentation(df)

    # Validate that the input dataframe is not changed
    def test_input_dataframe_not_changed(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()

        hml_segmentation = HMLSegmentation(base_df)
        _ = hml_segmentation.df

        assert original_df.equals(base_df)  # Check if the original dataframe is not changed

    def test_alternate_value_col(self, base_df):
        """Test that the method correctly segments a DataFrame with an alternate value column."""
        base_df = base_df.rename(columns={"total_price": "quantity"})
        hml_segmentation = HMLSegmentation(base_df, value_col="quantity")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"
