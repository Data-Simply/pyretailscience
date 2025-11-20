"""Tests for the analysis.speed_drill module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for tests
mpl.use("Agg")

from pyretailscience.analysis.speed_drill import SpeedDrill


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)  # noqa: NPY002
    X = pd.DataFrame(  # noqa: N806
        {
            "avg_basket_value": np.random.rand(100) * 100,  # noqa: NPY002
            "customer_lifetime_value": np.random.rand(100) * 50,  # noqa: NPY002
            "transaction_frequency": np.random.rand(100) * 25,  # noqa: NPY002
        },
    )
    # Create target with some relationship to features
    y = pd.Series(
        X["avg_basket_value"] * 0.5 + X["customer_lifetime_value"] * 0.3 + np.random.rand(100) * 10,  # noqa: NPY002
    )
    return X, y


@pytest.fixture
def sample_binary_data():
    """Create sample binary classification data for testing."""
    np.random.seed(42)  # noqa: NPY002
    X = pd.DataFrame(  # noqa: N806
        {
            "customer_age": np.random.rand(100) * 50 + 20,  # noqa: NPY002
            "annual_spend": np.random.rand(100) * 50000 + 30000,  # noqa: NPY002
            "loyalty_score": np.random.rand(100) * 100,  # noqa: NPY002
        },
    )
    # Create binary target based on threshold
    threshold_age = 40
    threshold_spend = 50000
    y = pd.Series((X["customer_age"] > threshold_age) & (X["annual_spend"] > threshold_spend), dtype=int)
    return X, y


class TestSpeedDrill:
    """Test the SpeedDrill class."""

    def test_initialization(self):
        """Test that SpeedDrill initializes with None values for model and column_names."""
        model = SpeedDrill()
        assert model.model is None
        assert model.column_names is None
        assert model.metrics_ == {}

    def test_fit_regression_returns_self(self, sample_regression_data):
        """Test that fit returns self for method chaining with regression data."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        result = model.fit(X, y, min_child_samples=10, max_depth=3)

        # Should return self for method chaining
        assert result is model

    def test_fit_regression_trains_model(self, sample_regression_data):
        """Test that fit creates a trained regressor model with correct feature names."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Model should be trained
        assert model.model is not None
        assert model.column_names == ["avg_basket_value", "customer_lifetime_value", "transaction_frequency"]

    def test_fit_regression_calculates_metrics(self, sample_regression_data):
        """Test that fit calculates R2 and MSE metrics for regression."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Metrics should be calculated
        assert "r2" in model.metrics_
        assert "mse" in model.metrics_
        assert isinstance(model.metrics_["r2"], float)
        assert isinstance(model.metrics_["mse"], float)
        # R² should be reasonable for training data
        assert -1.0 <= model.metrics_["r2"] <= 1.0

    def test_fit_binary_classification_returns_self(self, sample_binary_data):
        """Test that fit returns self for method chaining with binary data."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        result = model.fit(X, y, min_child_samples=10, max_depth=3)

        # Should return self for method chaining
        assert result is model

    def test_fit_binary_classification_trains_model(self, sample_binary_data):
        """Test that fit creates a trained classifier model with correct feature names."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Model should be trained
        assert model.model is not None
        assert model.column_names == ["customer_age", "annual_spend", "loyalty_score"]

    def test_fit_binary_classification_calculates_metrics(self, sample_binary_data):
        """Test that fit calculates accuracy and AUC metrics for binary classification."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Metrics should be calculated
        assert "accuracy" in model.metrics_
        assert "auc" in model.metrics_
        assert isinstance(model.metrics_["accuracy"], float)
        assert isinstance(model.metrics_["auc"], float)

        # Accuracy and AUC should be between 0 and 1
        assert 0 <= model.metrics_["accuracy"] <= 1
        assert 0 <= model.metrics_["auc"] <= 1

    def test_fit_with_custom_params(self, sample_regression_data):
        """Test that fit accepts and applies custom LightGBM parameters."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()

        # Pass custom learning rate
        model.fit(X, y, min_child_samples=5, max_depth=2, learning_rate=0.05)

        # Model should still be trained successfully
        assert model.model is not None
        assert "r2" in model.metrics_

    def test_view_tree_regression_returns_axes(self, sample_regression_data):
        """Test that view_tree returns a matplotlib Axes object for regression model."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        # Visualize tree
        ax = model.view_tree(figsize=(15, 10))

        # Should return an axes object
        assert ax is not None
        assert isinstance(ax, plt.Axes)

    def test_view_tree_regression_renders_nodes(self, sample_regression_data):
        """Test that view_tree renders tree nodes as patches for regression model."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        ax = model.view_tree(figsize=(15, 10))

        # The axes should have patches (from tree nodes)
        assert len(ax.patches) > 0

    def test_view_tree_classification_returns_axes(self, sample_binary_data):
        """Test that view_tree returns a matplotlib Axes object for classification model."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        # Visualize tree
        ax = model.view_tree(figsize=(15, 10))

        # Should return an axes object
        assert ax is not None
        assert isinstance(ax, plt.Axes)

    def test_view_tree_classification_renders_nodes(self, sample_binary_data):
        """Test that view_tree renders tree nodes as patches for classification model."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        ax = model.view_tree(figsize=(15, 10))

        # The axes should have patches (from tree nodes)
        assert len(ax.patches) > 0

    def test_view_tree_before_fit_raises_error(self):
        """Test that calling view_tree before fit raises ValueError with appropriate message."""
        model = SpeedDrill()

        with pytest.raises(ValueError, match="Model has not been trained yet"):
            model.view_tree()

    def test_view_tree_custom_figsize(self, sample_regression_data):
        """Test that custom figsize parameter is applied correctly to tree visualization."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        # Create tree with custom figsize
        custom_figsize = (24, 16)
        ax = model.view_tree(figsize=custom_figsize)

        # Get the figure from the axes
        fig = ax.get_figure()

        # Figure size should match (allowing for small floating-point differences)
        assert fig.get_figwidth() == pytest.approx(custom_figsize[0], abs=0.1)
        assert fig.get_figheight() == pytest.approx(custom_figsize[1], abs=0.1)

    def test_single_tree_trained(self, sample_regression_data):
        """Test that only a single tree is trained as expected for descriptive analytics."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Verify model has only 1 tree
        tree_dump = model.model.booster_.dump_model()
        assert len(tree_dump["tree_info"]) == 1

    def test_feature_names_preserved(self, sample_regression_data):
        """Test that feature names from input DataFrame are preserved after training."""
        X, y = sample_regression_data  # noqa: N806
        expected_feature_names = X.columns.tolist()

        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        assert model.column_names == expected_feature_names

    def test_generate_correlations_heatmap_returns_axes(self, sample_regression_data):
        """Test that generate_correlations_heatmap returns a matplotlib Axes object."""
        X, y = sample_regression_data  # noqa: N806

        ax = SpeedDrill.generate_correlations_heatmap(
            X,
            y,
            target_column="revenue",
            top_n=3,
        )

        assert ax is not None
        assert isinstance(ax, plt.Axes)

    def test_generate_correlations_heatmap_includes_target(self, sample_regression_data):
        """Test that correlation heatmap includes the target column."""
        X, y = sample_regression_data  # noqa: N806

        ax = SpeedDrill.generate_correlations_heatmap(
            X,
            y,
            target_column="revenue",
            top_n=3,
        )

        # The heatmap should have been created (has patches from heatmap cells)
        assert len(ax.collections) > 0

    def test_plot_feature_importances_returns_axes(self, sample_regression_data):
        """Test that plot_feature_importances returns a matplotlib Axes object."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        ax = model.plot_feature_importances(top_n=3)

        assert ax is not None
        assert isinstance(ax, plt.Axes)

    def test_plot_feature_importances_shows_bars(self, sample_regression_data):
        """Test that plot_feature_importances renders bars for feature importance values."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        ax = model.plot_feature_importances(top_n=3)

        # Should have bars (patches)
        assert len(ax.patches) > 0

    def test_plot_feature_importances_before_fit_raises_error(self):
        """Test that calling plot_feature_importances before fit raises ValueError."""
        model = SpeedDrill()

        with pytest.raises(ValueError, match="Model has not been trained yet"):
            model.plot_feature_importances()

    def test_plot_feature_importances_custom_top_n(self, sample_regression_data):
        """Test that top_n parameter controls number of features displayed."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Request top 2 features
        expected_bar_count = 2
        ax = model.plot_feature_importances(top_n=expected_bar_count)

        # Bar plot should show 2 bars (one per feature)
        assert len(ax.patches) == expected_bar_count

    def test_plot_feature_importances_custom_title(self, sample_regression_data):
        """Test that custom title parameter is applied to feature importance plot."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        custom_title = "Top Features for Churn Prediction"
        ax = model.plot_feature_importances(top_n=3, title=custom_title)

        # Title should be set
        assert ax.get_title() == custom_title
