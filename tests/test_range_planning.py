import numpy as np
import pandas as pd
import pyretailscience.range_planning as rp
import pytest


class TestCustomerDecisionHierarchy:
    def test_calculate_yules_q_identical_arrays(self):
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([1, 0, 1, 0, 1], dtype=bool)
        expected_q = 1.0

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_calculate_yules_q_opposite_arrays(self):
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([0, 1, 0, 1, 0], dtype=bool)
        expected_q = -1.0

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_calculate_yules_q_different_length_arrays(self):
        bought_product_1 = np.array([1, 0, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([1, 0, 1, 0], dtype=bool)

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2)

    def test_calculate_yules_q_empty_arrays(self):
        bought_product_1 = np.array([], dtype=bool)
        bought_product_2 = np.array([], dtype=bool)
        expected_q = 0.0

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_get_yules_q_distances(self):
        bought_product_1 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
        bought_product_2 = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        expected_q = -0.6363636363636364

        assert rp.CustomerDecisionHierarchy._calculate_yules_q(bought_product_1, bought_product_2) == expected_q

    def test_init_invalid_dataframe(self):
        df = pd.DataFrame({"customer_id": [1, 2, 3], "transaction_id": [1, 2, 3], "product_name": ["A", "B", "C"]})
        exclude_same_transaction_products = True
        random_state = 42

        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy(df, exclude_same_transaction_products, random_state)

    def test_init_exclude_same_transaction_products_true(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 2, 2, 2, 3, 3],
                "transaction_id": [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["A", "B", "C", "D", "E", "E", "E", "E"],
            }
        )
        exclude_same_transaction_products = True

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(df, exclude_same_transaction_products)

        expected_pairs_df = pd.DataFrame({"customer_id": [1, 3], "product_name": ["C", "E"]}).astype("category")

        assert pairs_df.equals(expected_pairs_df)

    def test_init_exclude_same_transaction_products_false(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 2, 2, 2, 3, 3],
                "transaction_id": [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["A", "B", "C", "D", "E", "E", "E", "E"],
            }
        )
        exclude_same_transaction_products = False

        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(df, exclude_same_transaction_products)

        expected_pairs_df = pd.DataFrame(
            {"customer_id": [1, 1, 1, 2, 2, 3], "product_name": ["A", "B", "C", "D", "E", "E"]}
        ).astype("category")

        assert pairs_df.equals(expected_pairs_df)
