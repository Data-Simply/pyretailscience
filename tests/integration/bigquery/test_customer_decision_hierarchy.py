"""Integration tests for Customer Decision Hierarchy Analysis with BigQuery."""

import numpy as np
import pandas as pd
import pytest

import pyretailscience.analysis.customer_decision_hierarchy as rp
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


class TestCustomerDecisionHierarchyBigQuery:
    """Integration tests for the CustomerDecisionHierarchy class using BigQuery data."""

    @pytest.fixture
    def sample_transactions_df(self, transactions_table):
        """Get a sample of transaction data from BigQuery for testing."""
        query = transactions_table.select(
            ["transaction_id", "customer_id", "product_name"],
        ).limit(1000)

        return query.execute()

    def test_calculate_yules_q_identical_arrays(self):
        """Test that the function returns 1.0 when the arrays are identical."""
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([1, 0, 1, 0, 1], dtype=bool)
        expected_q = 1.0

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_calculate_yules_q_opposite_arrays(self):
        """Test that the function returns -1.0 when the arrays are opposite."""
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([0, 1, 0, 1, 0], dtype=bool)
        expected_q = -1.0

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_calculate_yules_q_different_length_arrays(self):
        """Test that the function raises a ValueError when the arrays have different lengths."""
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([1, 0, 1, 0], dtype=bool)

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2)

    def test_calculate_yules_q_empty_arrays(self):
        """Test that the function returns 0.0 when the arrays are empty."""
        bought_product_1 = np.array([], dtype=bool)
        bought_product_2 = np.array([], dtype=bool)
        expected_q = 0.0

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_get_yules_q_distances(self):
        """Test that the function returns the correct Yules Q distances."""
        bought_product_1 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        expected_q = -0.6363636363636364

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_init_with_bigquery_data(self, sample_transactions_df):
        """Test initialization with data from BigQuery."""
        assert cols.customer_id in sample_transactions_df.columns
        assert cols.transaction_id in sample_transactions_df.columns
        assert "product_name" in sample_transactions_df.columns

        exclude_same_transaction_products = True
        random_state = 42

        cdh = rp.CustomerDecisionHierarchy(
            sample_transactions_df,
            product_col="product_name",
            exclude_same_transaction_products=exclude_same_transaction_products,
            random_state=random_state,
        )

        assert cdh is not None
        assert isinstance(cdh, rp.CustomerDecisionHierarchy)

    def test_init_invalid_dataframe(self):
        """Test that the function raises a ValueError when the dataframe is invalid."""
        df = pd.DataFrame(
            {cols.customer_id: [1, 2, 3], cols.transaction_id: [1, 2, 3]},  # Missing product_name column
        )
        exclude_same_transaction_products = True
        random_state = 42

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy(df, exclude_same_transaction_products, random_state)

    def test_get_pairs_with_bigquery_data(self, sample_transactions_df):
        """Test that the function returns the correct pairs dataframe using BigQuery data."""
        exclude_same_transaction_products = True

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            sample_transactions_df,
            exclude_same_transaction_products=exclude_same_transaction_products,
            product_col="product_name",
        )

        assert cols.customer_id in pairs_df.columns
        assert "product_name" in pairs_df.columns

        if len(pairs_df) > 0:
            merged_with_transactions = pairs_df.merge(
                sample_transactions_df,
                on=[cols.customer_id, "product_name"],
            )

            assert len(merged_with_transactions) > 0
