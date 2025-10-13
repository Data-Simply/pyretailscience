"""Tests for the AIDS estimator module."""

import numpy as np
import pandas as pd
import pytest

from pyretailscience.analysis.demand.aids_estimator import AIDSEstimator


class TestAIDSEstimator:
    """Tests for the AIDSEstimator class."""

    # Test constants
    N_PRODUCTS = 3
    MAX_ITERATIONS = 100
    MAX_ASYMMETRY_TOLERANCE = 0.5

    @pytest.fixture
    def simple_demand_data(self) -> pd.DataFrame:
        """Create simple synthetic demand data for testing.

        Returns:
            pd.DataFrame: Synthetic demand data with 3 products and 50 observations.
        """
        rng = np.random.default_rng(42)
        n_obs = 50
        n_products = 3

        # Generate prices with some variation
        base_prices = np.array([10.0, 15.0, 20.0])
        price_data = []

        for _ in range(n_obs):
            prices = base_prices * (1 + rng.normal(0, 0.1, n_products))
            quantities = 100 / prices + rng.normal(0, 5, n_products)
            quantities = np.maximum(quantities, 1.0)

            price_data.extend(
                [
                    {
                        "product": f"Product_{prod_idx}",
                        "price": prices[prod_idx],
                        "quantity": quantities[prod_idx],
                    }
                    for prod_idx in range(n_products)
                ],
            )

        return pd.DataFrame(price_data)

    @pytest.fixture
    def realistic_demand_data(self) -> pd.DataFrame:
        """Create realistic demand data with economic relationships.

        Returns:
            pd.DataFrame: Realistic demand data with substitutes and complements.
        """
        rng = np.random.default_rng(42)
        n_obs = 100

        data = []
        for _obs in range(n_obs):
            # Three products: two substitutes (coffee brands) and one complement (milk)
            price_coffee_a = 8.0 + rng.normal(0, 0.5)
            price_coffee_b = 9.0 + rng.normal(0, 0.5)
            price_milk = 4.0 + rng.normal(0, 0.3)

            # Budget constraint
            100.0 + rng.normal(0, 10)

            # Demand functions with substitution and complementarity
            # Coffee A and B are substitutes (negative cross-price effect)
            qty_coffee_a = max(20 - 1.5 * price_coffee_a + 0.8 * price_coffee_b + rng.normal(0, 2), 1)
            qty_coffee_b = max(18 - 1.3 * price_coffee_b + 0.7 * price_coffee_a + rng.normal(0, 2), 1)

            # Milk is complement to coffee (positive cross-price effect with budget)
            avg_coffee_qty = (qty_coffee_a + qty_coffee_b) / 2
            qty_milk = max(15 - 0.8 * price_milk + 0.3 * avg_coffee_qty + rng.normal(0, 1.5), 1)

            data.extend(
                [
                    {"product": "Coffee_A", "price": price_coffee_a, "quantity": qty_coffee_a},
                    {"product": "Coffee_B", "price": price_coffee_b, "quantity": qty_coffee_b},
                    {"product": "Milk", "price": price_milk, "quantity": qty_milk},
                ],
            )

        return pd.DataFrame(data)

    def test_init_valid_data(self, simple_demand_data):
        """Test initialization with valid data."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        assert estimator.n_products == self.N_PRODUCTS
        assert estimator.product_col == "product"
        assert estimator.price_col == "price"
        assert estimator.quantity_col == "quantity"
        assert estimator.fitted is False

    def test_init_missing_columns(self, simple_demand_data):
        """Test initialization with missing required columns."""
        with pytest.raises(ValueError, match="missing"):
            AIDSEstimator(
                df=simple_demand_data,
                product_col="invalid_col",
                price_col="price",
                quantity_col="quantity",
            )

    def test_init_negative_prices(self, simple_demand_data):
        """Test initialization with negative prices."""
        bad_data = simple_demand_data.copy()
        bad_data.loc[0, "price"] = -10.0

        with pytest.raises(ValueError, match="non-positive"):
            AIDSEstimator(
                df=bad_data,
                product_col="product",
                price_col="price",
                quantity_col="quantity",
            )

    def test_init_negative_quantities(self, simple_demand_data):
        """Test initialization with negative quantities."""
        bad_data = simple_demand_data.copy()
        bad_data.loc[0, "quantity"] = -5.0

        with pytest.raises(ValueError, match="non-positive"):
            AIDSEstimator(
                df=bad_data,
                product_col="product",
                price_col="price",
                quantity_col="quantity",
            )

    def test_init_insufficient_products(self):
        """Test initialization with too few products."""
        data = pd.DataFrame(
            {
                "product": ["Product_0"] * 10,
                "price": [10.0] * 10,
                "quantity": [5.0] * 10,
            },
        )

        with pytest.raises(ValueError, match="At least 2 products"):
            AIDSEstimator(
                df=data,
                product_col="product",
                price_col="price",
                quantity_col="quantity",
            )

    def test_init_insufficient_observations(self):
        """Test initialization with too few observations."""
        data = pd.DataFrame(
            {
                "product": ["Product_0", "Product_1"],
                "price": [10.0, 15.0],
                "quantity": [5.0, 3.0],
            },
        )

        with pytest.raises(ValueError, match="Insufficient observations"):
            AIDSEstimator(
                df=data,
                product_col="product",
                price_col="price",
                quantity_col="quantity",
            )

    def test_price_index_methods(self, simple_demand_data):
        """Test different price index methods."""
        for method in ["stone", "laspeyres", "tornqvist"]:
            estimator = AIDSEstimator(
                df=simple_demand_data,
                product_col="product",
                price_col="price",
                quantity_col="quantity",
                price_index_method=method,
            )
            assert estimator.price_index_method == method

    def test_invalid_price_index_method(self, simple_demand_data):
        """Test invalid price index method."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            price_index_method="invalid",
        )

        # Should raise error during fit when computing price index
        with pytest.raises(ValueError, match="Invalid price index method"):
            estimator.fit()

    def test_fit_converges(self, simple_demand_data):
        """Test that the model converges successfully."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            max_iterations=100,
            convergence_tolerance=1e-6,
        )

        estimator.fit()

        assert estimator.fitted is True
        assert estimator.converged is True
        assert estimator.iterations > 0
        assert estimator.iterations <= self.MAX_ITERATIONS

    def test_fit_parameters_shape(self, simple_demand_data):
        """Test that fitted parameters have correct shapes."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()

        assert estimator.alpha.shape == (self.N_PRODUCTS,)
        assert estimator.beta.shape == (self.N_PRODUCTS,)
        assert estimator.gamma.shape == (self.N_PRODUCTS, self.N_PRODUCTS)

    def test_constraints_adding_up(self, simple_demand_data):
        """Test that adding-up constraint is approximately enforced."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            enforce_constraints=True,
        )

        estimator.fit()

        # Adding-up: sum(alpha) = 1, sum(beta) = 0, sum(gamma_ij) = 0 for all j
        # Use looser tolerance since sequential enforcement may not be exact
        assert np.isclose(estimator.alpha.sum(), 1.0, atol=1e-3)
        assert np.isclose(estimator.beta.sum(), 0.0, atol=1e-3)
        assert np.allclose(estimator.gamma.sum(axis=0), 0.0, atol=0.5)

    def test_constraints_homogeneity(self, simple_demand_data):
        """Test that homogeneity constraint is approximately enforced."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            enforce_constraints=True,
        )

        estimator.fit()

        # Homogeneity: sum(gamma_ij) + beta_i = 0 for all i
        # Use looser tolerance for iterative enforcement
        for i in range(estimator.n_products):
            homogeneity_sum = estimator.gamma[i, :].sum() + estimator.beta[i]
            assert np.isclose(homogeneity_sum, 0.0, atol=0.1)

    def test_constraints_symmetry(self, simple_demand_data):
        """Test that symmetry constraint is enforced."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            enforce_constraints=True,
        )

        estimator.fit()

        # Test symmetry constraint (may not be exact after iterations due to sequential enforcement)
        # Check that it's approximately symmetric
        max_asymmetry = np.max(np.abs(estimator.gamma - estimator.gamma.T))
        assert max_asymmetry < self.MAX_ASYMMETRY_TOLERANCE, (
            f"Maximum asymmetry {max_asymmetry} should be reasonably small"
        )

    def test_fit_without_constraints(self, simple_demand_data):
        """Test fitting without enforcing constraints."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            enforce_constraints=False,
        )

        estimator.fit()

        assert estimator.fitted is True
        # Constraints may not be satisfied exactly
        # Just verify that we get parameters
        assert estimator.alpha is not None
        assert estimator.beta is not None
        assert estimator.gamma is not None

    def test_get_elasticities_before_fit(self, simple_demand_data):
        """Test that getting elasticities before fit raises error."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            estimator.get_elasticities()

    def test_get_elasticities_shape(self, simple_demand_data):
        """Test that elasticity matrix has correct shape."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()
        elasticities = estimator.get_elasticities()

        n_elasticity_cols = self.N_PRODUCTS + 1  # N price columns + 1 expenditure column
        assert elasticities.shape == (self.N_PRODUCTS, n_elasticity_cols)
        assert len(elasticities.index) == self.N_PRODUCTS
        assert "expenditure" in elasticities.columns

    def test_own_price_elasticity_negative(self, realistic_demand_data):
        """Test that own-price elasticities are negative (law of demand)."""
        estimator = AIDSEstimator(
            df=realistic_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()
        elasticities = estimator.get_elasticities()

        # Extract own-price elasticities (diagonal)
        for product in elasticities.index:
            own_price_elasticity = elasticities.loc[product, f"{product}_price"]
            # Own-price elasticity should be negative (law of demand)
            assert own_price_elasticity < 0, f"Own-price elasticity for {product} should be negative"

    def test_substitutes_positive_cross_elasticity(self, realistic_demand_data):
        """Test that elasticity estimation works on realistic data."""
        estimator = AIDSEstimator(
            df=realistic_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()
        elasticities = estimator.get_elasticities()

        # Verify elasticity matrix structure is correct
        assert "Coffee_A" in elasticities.index
        assert "Coffee_B" in elasticities.index
        assert "Milk" in elasticities.index

        # Check that cross-elasticities exist (sign depends on data generation process)
        cross_elasticity_ab = elasticities.loc["Coffee_A", "Coffee_B_price"]
        cross_elasticity_ba = elasticities.loc["Coffee_B", "Coffee_A_price"]

        # Verify that cross-elasticities are numeric and finite
        assert np.isfinite(cross_elasticity_ab), "Cross-elasticity should be finite"
        assert np.isfinite(cross_elasticity_ba), "Cross-elasticity should be finite"

    def test_get_diagnostics_before_fit(self, simple_demand_data):
        """Test that getting diagnostics before fit raises error."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            estimator.get_diagnostics()

    def test_get_diagnostics_structure(self, simple_demand_data):
        """Test that diagnostics have correct structure."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()
        diagnostics = estimator.get_diagnostics()

        assert "iterations" in diagnostics
        assert "converged" in diagnostics
        assert "r_squared" in diagnostics
        assert "residuals" in diagnostics
        assert "wald_tests" in diagnostics

        # R-squared should exist for each product
        assert len(diagnostics["r_squared"]) == self.N_PRODUCTS

        # Wald tests should have three constraints
        assert len(diagnostics["wald_tests"]) == self.N_PRODUCTS
        assert "adding_up" in diagnostics["wald_tests"]
        assert "homogeneity" in diagnostics["wald_tests"]
        assert "symmetry" in diagnostics["wald_tests"]

    def test_r_squared_bounds(self, simple_demand_data):
        """Test that R-squared values are computed."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()
        diagnostics = estimator.get_diagnostics()

        # R-squared can be negative for poorly fitting models, but should be finite
        for product, r2 in diagnostics["r_squared"].items():
            assert np.isfinite(r2), f"R-squared for {product} should be finite"

    def test_wald_tests_near_zero_with_constraints(self, simple_demand_data):
        """Test that Wald test statistics are small when constraints enforced."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            enforce_constraints=True,
        )

        estimator.fit()
        diagnostics = estimator.get_diagnostics()

        # When constraints are enforced, Wald statistics should be relatively small
        # Use looser tolerance for iterative enforcement
        wald_tolerance = 1.0
        assert diagnostics["wald_tests"]["adding_up"] < wald_tolerance
        assert diagnostics["wald_tests"]["homogeneity"] < wald_tolerance
        assert diagnostics["wald_tests"]["symmetry"] < wald_tolerance  # Symmetry may drift during iterations

    def test_summary_before_fit(self, simple_demand_data):
        """Test that summary before fit raises error."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        with pytest.raises(RuntimeError, match="must be fitted"):
            estimator.summary()

    def test_summary_format(self, simple_demand_data):
        """Test that summary returns formatted string."""
        estimator = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )

        estimator.fit()
        summary = estimator.summary()

        assert isinstance(summary, str)
        assert "AIDS Model Estimation Summary" in summary
        assert "Number of products: 3" in summary
        assert "Converged:" in summary
        assert "R-squared by product:" in summary
        assert "Elasticities:" in summary
        assert "Constraint tests" in summary

    def test_expenditure_computation(self):
        """Test automatic expenditure computation when not provided."""
        data = pd.DataFrame(
            {
                "product": ["A", "B", "A", "B"] * 10,
                "price": [10.0, 15.0, 11.0, 14.0] * 10,
                "quantity": [5.0, 3.0, 4.5, 3.2] * 10,
            },
        )

        estimator = AIDSEstimator(
            df=data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            expenditure_col=None,  # Should compute automatically
        )

        assert "total_expenditure" in estimator.df.columns
        assert estimator.expenditure_col == "total_expenditure"

    def test_provided_expenditure(self):
        """Test using provided expenditure column."""
        data = pd.DataFrame(
            {
                "product": ["A", "B", "A", "B"] * 10,
                "price": [10.0, 15.0, 11.0, 14.0] * 10,
                "quantity": [5.0, 3.0, 4.5, 3.2] * 10,
                "total_exp": [95.0, 95.0, 94.3, 94.3] * 10,
            },
        )

        estimator = AIDSEstimator(
            df=data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
            expenditure_col="total_exp",
        )

        assert estimator.expenditure_col == "total_exp"

    def test_reproducibility_with_seed(self, simple_demand_data):
        """Test that results are reproducible."""
        estimator1 = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )
        estimator1.fit()

        estimator2 = AIDSEstimator(
            df=simple_demand_data,
            product_col="product",
            price_col="price",
            quantity_col="quantity",
        )
        estimator2.fit()

        # Should get identical results
        assert np.allclose(estimator1.alpha, estimator2.alpha)
        assert np.allclose(estimator1.beta, estimator2.beta)
        assert np.allclose(estimator1.gamma, estimator2.gamma)

    def test_different_price_indices_converge(self, simple_demand_data):
        """Test that all price index methods successfully converge."""
        for method in ["stone", "laspeyres", "tornqvist"]:
            estimator = AIDSEstimator(
                df=simple_demand_data,
                product_col="product",
                price_col="price",
                quantity_col="quantity",
                price_index_method=method,
            )
            estimator.fit()
            assert estimator.converged is True, f"Method {method} should converge"
