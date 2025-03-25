"""Tests for the RFMSegmentation class."""

import pandas as pd
import pytest
from freezegun import freeze_time

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.rfm import RFMSegmentation

cols = ColumnHelper()


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

    @freeze_time("2025-03-19")
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
