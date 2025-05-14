"""Integration tests for the RFMSegmentation class using BigQuery."""

import pandas as pd
import pytest
from freezegun import freeze_time

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.rfm import RFMSegmentation

cols = ColumnHelper()

REASONABLE_THRESHOLD = 0.1


class TestRFMSegmentationBigQuery:
    """Integration tests for the RFMSegmentation class using BigQuery."""

    @pytest.fixture
    def transactions_df(self, transactions_table):
        """Return a DataFrame from BigQuery for testing."""
        query = transactions_table.select(
            transactions_table.transaction_id,
            transactions_table.transaction_date,
            transactions_table.customer_id,
            transactions_table.unit_spend,
        )

        return query.execute()

    @pytest.fixture
    def expected_df(self):
        """Returns the expected DataFrame structure for testing segmentation."""
        return pd.DataFrame(
            columns=[
                "recency_days",
                "frequency",
                "monetary",
                "r_score",
                "f_score",
                "m_score",
                "rfm_segment",
                "fm_segment",
            ],
        )

    def test_rfm_segmentation_with_bigquery_data(self, transactions_df):
        """Test that the RFM segmentation correctly processes BigQuery data."""
        current_date = "2025-03-17"

        assert not transactions_df.empty, "No data was returned from BigQuery"

        required_columns = [cols.customer_id, cols.transaction_id, cols.unit_spend, cols.transaction_date]
        for col in required_columns:
            assert col in transactions_df.columns, f"Column {col} is missing from the DataFrame"

        rfm_segmentation = RFMSegmentation(df=transactions_df, current_date=current_date)
        result_df = rfm_segmentation.df

        assert not result_df.empty, "Segmentation produced empty results"
        assert "rfm_segment" in result_df.columns, "RFM segment column not found in results"
        assert all(
            c in result_df.columns for c in ["recency_days", "frequency", "monetary", "r_score", "f_score", "m_score"]
        ), "Missing expected columns in result"

    def test_handles_bigquery_connection_with_multiple_transactions(self, transactions_table):
        """Test that the method handles multiple transactions per customer from BigQuery."""
        query = (
            transactions_table.group_by(transactions_table.customer_id)
            .aggregate(transaction_count=transactions_table.transaction_id.count())
            .limit(10)
        )

        all_customers = query.execute()

        customers_with_multiple = all_customers[all_customers["transaction_count"] > 1]

        if customers_with_multiple.empty:
            pytest.skip("No customers with multiple transactions found in test data")

        customer_ids = customers_with_multiple.customer_id.tolist()

        query = transactions_table.filter(
            transactions_table.customer_id.isin(customer_ids),
        ).select(
            transactions_table.transaction_id,
            transactions_table.transaction_date,
            transactions_table.customer_id,
            transactions_table.unit_spend,
        )

        transactions_df = query.execute()

        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=transactions_df, current_date=current_date)
        result_df = rfm_segmentation.df

        assert len(result_df) <= len(customers_with_multiple), "Number of segmented customers exceeds expected count"

        for customer_id in result_df.index:
            actual_frequency = result_df.loc[customer_id, "frequency"]
            assert actual_frequency >= 1, f"Customer {customer_id} should have at least one transaction"

            customer_transactions = transactions_df[transactions_df[cols.customer_id] == customer_id]
            raw_transaction_count = len(customer_transactions)

            ratio = max(actual_frequency, 1) / max(raw_transaction_count, 1)
            inverse_ratio = max(raw_transaction_count, 1) / max(actual_frequency, 1)
            reasonable = min(ratio, inverse_ratio) >= REASONABLE_THRESHOLD

            assert reasonable, (
                f"Frequency value for customer {customer_id}: {actual_frequency} is too far from raw transaction count: {raw_transaction_count}"
            )

    @freeze_time("2025-03-19")
    def test_rfm_segmentation_with_no_date_bigquery(self, transactions_df):
        """Test that the RFM segmentation works with automatic current date."""
        rfm_segmentation = RFMSegmentation(df=transactions_df)
        result_df = rfm_segmentation.df

        current_date = pd.Timestamp("2025-03-19")

        sample_customers = result_df.head(3).index.tolist()
        for customer_id in sample_customers:
            customer_transactions = transactions_df[transactions_df[cols.customer_id] == customer_id].copy()
            customer_transactions[cols.transaction_date] = pd.to_datetime(customer_transactions[cols.transaction_date])
            latest_transaction_date = customer_transactions[cols.transaction_date].max()

            expected_recency = (current_date - latest_transaction_date).days
            actual_recency = result_df.loc[customer_id, "recency_days"]

            assert abs(actual_recency - expected_recency) <= 1, (
                f"Recency days mismatch for customer {customer_id}: expected {expected_recency}, got {actual_recency}"
            )

    def test_ibis_table_direct_usage(self, transactions_table, bigquery_connection):
        """Test that RFMSegmentation works directly with an Ibis table."""
        ibis_table = transactions_table.select(
            transactions_table.transaction_id,
            transactions_table.transaction_date,
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(100)

        pandas_df = ibis_table.execute()

        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=pandas_df, current_date=current_date)
        result_df = rfm_segmentation.df

        assert not result_df.empty, "Segmentation produced empty results"
        assert "rfm_segment" in result_df.columns, "RFM segment column not found in results"

        expected_columns = [
            "recency_days",
            "frequency",
            "monetary",
            "r_score",
            "f_score",
            "m_score",
            "rfm_segment",
            "fm_segment",
        ]
        assert all(column in result_df.columns for column in expected_columns), (
            f"Missing columns in result DataFrame. Expected: {expected_columns}, Got: {result_df.columns.tolist()}"
        )

    def test_large_dataset_performance(self, transactions_table):
        """Test RFMSegmentation performance with a larger dataset."""
        query = transactions_table.select(
            transactions_table.transaction_id,
            transactions_table.transaction_date,
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(10000)

        large_df = query.execute()

        rfm_segmentation = RFMSegmentation(df=large_df, current_date="2025-03-17")
        result_df = rfm_segmentation.df

        assert not result_df.empty, "Large dataset segmentation produced empty results"
        assert len(result_df) <= len(large_df[cols.customer_id].unique()), (
            "More segmented customers than unique customers in dataset"
        )
