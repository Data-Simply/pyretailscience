"""Tests for the customer_decision_hierarchy module."""

import numpy as np
import pandas as pd
import pytest

import pyretailscience.analysis.customer_decision_hierarchy as rp
from pyretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()


class TestCustomerDecisionHierarchy:
    """Tests for the CustomerDecisionHierarchy class."""

    # ========================================================================
    # FIXTURES - Reusable test data
    # ========================================================================

    @pytest.fixture
    def simple_transaction_data(self):
        """Simple transaction data for basic Yule's Q tests."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3],
                cols.transaction_id: [1, 1, 2, 3, 3, 4, 5, 6],
                "product_name": ["A", "B", "C", "D", "E", "E", "E", "E"],
            },
        )

    @pytest.fixture
    def aids_test_data(self):
        """Standard test data for AIDS method with price and quantity."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                cols.transaction_id: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
                "price": [1.0, 1.2, 0.9, 1.1, 1.15, 0.95, 0.95, 1.25, 0.85, 1.05, 1.18, 0.92, 1.02, 1.22, 0.88],
                "quantity": [10, 8, 12, 9, 7, 11, 11, 9, 13, 8, 6, 10, 10, 8, 12],
            },
        )

    @pytest.fixture
    def aids_hierarchy(self, aids_test_data):
        """Pre-initialized AIDS hierarchy for testing."""
        return rp.CustomerDecisionHierarchy(
            df=aids_test_data,
            product_col="product",
            method="aids",
            price_col="price",
            quantity_col="quantity",
        )

    # ========================================================================
    # HELPER METHODS - Reusable assertions
    # ========================================================================

    def assert_valid_distance_matrix(self, distances, n_products):
        """Assert distance matrix has expected properties.

        Args:
            distances (np.ndarray): Distance matrix to validate.
            n_products (int): Expected number of products.
        """
        assert distances.shape == (n_products, n_products), "Distance matrix should be square"
        assert np.all(np.diag(distances) == 0), "Diagonal should be zero (distance to self)"
        assert np.all(distances >= 0), "All distances should be non-negative"
        assert np.all(distances <= 1), "All distances should be normalized to [0, 1]"

    # ========================================================================
    # YULE'S Q CALCULATION TESTS
    # ========================================================================

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

    # ========================================================================
    # INITIALIZATION AND BASIC FUNCTIONALITY TESTS
    # ========================================================================

    def test_init_invalid_dataframe(self, simple_transaction_data):
        """Test that the function raises a ValueError when the dataframe is invalid."""
        with pytest.raises(ValueError):
            rp.CustomerDecisionHierarchy(simple_transaction_data, "invalid_product_col", True)

    def test_init_exclude_same_transaction_products_true(self, simple_transaction_data):
        """Test that the function returns the correct pairs dataframe when exclude_same_transaction_products is True."""
        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            simple_transaction_data,
            exclude_same_transaction_products=True,
            product_col="product_name",
        )

        expected_pairs_df = pd.DataFrame({cols.customer_id: [1, 3], "product_name": ["C", "E"]}).astype("category")

        assert pairs_df.equals(expected_pairs_df)

    def test_init_exclude_same_transaction_products_false(self, simple_transaction_data):
        """Test that the function returns the correct pairs dataframe when exclude_same_transaction_products is False."""
        pairs_df = rp.CustomerDecisionHierarchy._get_pairs(
            simple_transaction_data,
            exclude_same_transaction_products=False,
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

    # ========================================================================
    # AIDS METHOD TESTS
    # ========================================================================

    def test_aids_method_initialization(self, aids_test_data):
        """Test that CustomerDecisionHierarchy can be initialized with AIDS method."""
        hierarchy = rp.CustomerDecisionHierarchy(
            df=aids_test_data,
            product_col="product",
            method="aids",
            price_col="price",
            quantity_col="quantity",
        )

        assert hasattr(hierarchy, "aids_estimator"), "Should create AIDS estimator"
        assert hierarchy.aids_estimator is not None, "AIDS estimator should be fitted"
        assert hierarchy.aids_estimator.fitted, "AIDS estimator should be fitted"
        assert hasattr(hierarchy, "distances"), "Should calculate distances"
        assert hierarchy.distances is not None, "Distances should not be None"

    def test_aids_method_missing_price_column(self, aids_test_data):
        """Test that AIDS method raises error when price column is missing."""
        df_no_price = aids_test_data.drop(columns=["price"])

        with pytest.raises(ValueError, match="price_col"):
            rp.CustomerDecisionHierarchy(
                df=df_no_price,
                product_col="product",
                method="aids",
                quantity_col="quantity",
            )

    def test_aids_method_missing_quantity_column(self, aids_test_data):
        """Test that AIDS method raises error when quantity column is missing."""
        df_no_quantity = aids_test_data.drop(columns=["quantity"])

        with pytest.raises(ValueError, match="quantity_col"):
            rp.CustomerDecisionHierarchy(
                df=df_no_quantity,
                product_col="product",
                method="aids",
                price_col="price",
            )

    def test_aids_method_distances_shape(self, aids_hierarchy, aids_test_data):
        """Test that AIDS method produces correct distance matrix shape."""
        n_products = aids_test_data["product"].nunique()
        self.assert_valid_distance_matrix(aids_hierarchy.distances, n_products)

    def test_aids_method_distances_normalized(self, aids_hierarchy):
        """Test that AIDS method produces normalized distances in [0, 1] range."""
        assert np.all(aids_hierarchy.distances >= 0), "All distances should be >= 0"
        assert np.all(aids_hierarchy.distances <= 1), "All distances should be <= 1"

    def test_aids_method_elasticities_computed(self, aids_hierarchy):
        """Test that AIDS method computes elasticities correctly."""
        elasticities = aids_hierarchy.aids_estimator.get_elasticities()
        n_products = 3  # Test data has 3 products (A, B, C)

        assert elasticities is not None, "Elasticities should be computed"
        assert len(elasticities) == n_products, f"Should have elasticities for {n_products} products"
        assert "expenditure" in elasticities.columns, "Should include expenditure elasticity"

        # Check own-price elasticities are negative (law of demand)
        own_price_elasticities = np.diag(elasticities.iloc[:, :n_products].values)
        assert np.all(own_price_elasticities < 0), "Own-price elasticities should be negative"

    def test_aids_method_works_independently(self, aids_hierarchy):
        """Test that AIDS method works and produces valid results."""
        # Comprehensive validation using helper method
        self.assert_valid_distance_matrix(aids_hierarchy.distances, 3)
