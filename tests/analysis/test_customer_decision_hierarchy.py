"""Tests for the customer_decision_hierarchy module."""

import numpy as np
import pandas as pd
import pytest

import pyretailscience.analysis.customer_decision_hierarchy as rp
from pyretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()


class TestCustomerDecisionHierarchy:
    """Tests for the CustomerDecisionHierarchy class."""

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

    def test_init_invalid_dataframe(self):
        """Test that the function raises a ValueError when the dataframe is invalid."""
        df = pd.DataFrame(
            {cols.customer_id: [1, 2, 3], cols.transaction_id: [1, 2, 3], "product_name": ["A", "B", "C"]},
        )
        exclude_same_transaction_products = True

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy(df, "invalid_product_col", exclude_same_transaction_products)

    def test_init_exclude_same_transaction_products_true(self):
        """Test that the function returns the correct pairs dataframe when exclude_same_transaction_products is True."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["A", "B", "C", "D", "E", "E", "E", "E"],
            },
        )
        exclude_same_transaction_products = True

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            df,
            exclude_same_transaction_products,
            product_col="product_name",
        )

        expected_pairs_df = pd.DataFrame({cols.customer_id: [1, 3], "product_name": ["C", "E"]}).astype("category")

        assert pairs_df.equals(expected_pairs_df)

    def test_init_exclude_same_transaction_products_false(self):
        """Test that the function returns the correct pairs dataframe when exclude_same_transaction_products is False."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["A", "B", "C", "D", "E", "E", "E", "E"],
            },
        )
        exclude_same_transaction_products = False

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            df,
            exclude_same_transaction_products,
            product_col="product_name",
        )

        expected_pairs_df = pd.DataFrame(
            {cols.customer_id: [1, 1, 1, 2, 2, 3], "product_name": ["A", "B", "C", "D", "E", "E"]},
        ).astype("category")

        assert pairs_df.equals(expected_pairs_df)

    def test_with_custom_column_names(self):
        """Test CustomerDecisionHierarchy with custom column names to ensure column overrides work correctly."""
        custom_test_df = pd.DataFrame(
            {
                "cust_identifier": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                "txn_identifier": [101, 102, 201, 202, 301, 302, 401, 402, 501, 502, 601, 602, 701, 702, 801, 802],
                "product_name": ["A", "B", "A", "C", "B", "C", "A", "D", "B", "D", "C", "D", "A", "B", "C", "D"],
            },
        )

        with option_context("column.customer_id", "cust_identifier", "column.transaction_id", "txn_identifier"):
            hierarchy = rp.CustomerDecisionHierarchy(
                df=custom_test_df,
                product_col="product_name",
                method="yules_q",
            )

            assert hasattr(hierarchy, "pairs_df"), "Should create pairs_df with custom columns"
            assert hasattr(hierarchy, "distances"), "Should create distances with custom columns"
            assert not hierarchy.pairs_df.empty, "Should produce results with custom column names"

            assert "cust_identifier" in hierarchy.pairs_df.columns, "Should handle custom customer_id column name"
            assert "product_name" in hierarchy.pairs_df.columns, "Should handle product column"
