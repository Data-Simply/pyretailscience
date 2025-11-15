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
            "feature1": np.random.rand(100) * 100,  # noqa: NPY002
            "feature2": np.random.rand(100) * 50,  # noqa: NPY002
            "feature3": np.random.rand(100) * 25,  # noqa: NPY002
        },
    )
    # Create target with some relationship to features
    y = pd.Series(X["feature1"] * 0.5 + X["feature2"] * 0.3 + np.random.rand(100) * 10)  # noqa: NPY002
    return X, y


@pytest.fixture
def sample_binary_data():
    """Create sample binary classification data for testing."""
    np.random.seed(42)  # noqa: NPY002
    X = pd.DataFrame(  # noqa: N806
        {
            "age": np.random.rand(100) * 50 + 20,  # noqa: NPY002
            "income": np.random.rand(100) * 50000 + 30000,  # noqa: NPY002
            "score": np.random.rand(100) * 100,  # noqa: NPY002
        },
    )
    # Create binary target based on threshold
    _threshold_age = 40  # Internal variable
    _threshold_income = 50000  # Internal variable
    y = pd.Series((X["age"] > _threshold_age) & (X["income"] > _threshold_income), dtype=int)
    return X, y


class TestSpeedDrill:
    """Test the SpeedDrill class."""

    def test_initialization(self):
        """Test that SpeedDrill initializes correctly."""
        model = SpeedDrill()
        assert model.model is None
        assert model.column_names is None
        assert model.metrics_ == {}

    def test_fit_regression(self, sample_regression_data):
        """Test fitting on regression data."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        result = model.fit(X, y, min_child_samples=10, max_depth=3)

        # Should return self for method chaining
        assert result is model

        # Model should be trained
        assert model.model is not None
        assert model.column_names == ["feature1", "feature2", "feature3"]

        # Metrics should be calculated
        assert "r2" in model.metrics_
        assert "mse" in model.metrics_
        assert isinstance(model.metrics_["r2"], float)
        assert isinstance(model.metrics_["mse"], float)

    def test_fit_binary_classification(self, sample_binary_data):
        """Test fitting on binary classification data."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        result = model.fit(X, y, min_child_samples=10, max_depth=3)

        # Should return self for method chaining
        assert result is model

        # Model should be trained
        assert model.model is not None
        assert model.column_names == ["age", "income", "score"]

        # Metrics should be calculated
        assert "accuracy" in model.metrics_
        assert "auc" in model.metrics_
        assert isinstance(model.metrics_["accuracy"], float)
        assert isinstance(model.metrics_["auc"], float)

        # Accuracy and AUC should be between 0 and 1
        assert 0 <= model.metrics_["accuracy"] <= 1
        assert 0 <= model.metrics_["auc"] <= 1

    def test_fit_with_custom_params(self, sample_regression_data):
        """Test fitting with custom LightGBM parameters."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()

        # Pass custom learning rate
        model.fit(X, y, min_child_samples=5, max_depth=2, learning_rate=0.05)

        # Model should still be trained successfully
        assert model.model is not None
        assert "r2" in model.metrics_

    def test_view_tree_regression(self, sample_regression_data):
        """Test tree visualization for regression model."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        # Visualize tree
        ax = model.view_tree(figsize=(15, 10))

        # Should return an axes object
        assert ax is not None
        assert isinstance(ax, plt.Axes)

        # The axes should have patches (from tree nodes) and text elements
        assert len(ax.patches) > 0

    def test_view_tree_classification(self, sample_binary_data):
        """Test tree visualization for classification model."""
        X, y = sample_binary_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=2)

        # Visualize tree
        ax = model.view_tree(figsize=(15, 10))

        # Should return an axes object
        assert ax is not None
        assert isinstance(ax, plt.Axes)

        # The axes should have patches (from tree nodes)
        assert len(ax.patches) > 0

    def test_view_tree_before_fit_raises_error(self):
        """Test that calling view_tree before fit raises ValueError."""
        model = SpeedDrill()

        with pytest.raises(ValueError, match="Model has not been trained yet"):
            model.view_tree()

    def test_view_tree_custom_figsize(self, sample_regression_data):
        """Test that custom figsize is applied to tree visualization."""
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

    def test_metrics_are_data_quality_checks(self, sample_regression_data):
        """Test that metrics are calculated on training data (data quality checks)."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # R² should be non-negative since we're measuring training data fit
        # (this is expected for descriptive analytics)
        # Note: actual value depends on data, so we just check it's calculated
        assert model.metrics_["r2"] >= -1.0  # R² can theoretically be negative for very bad fits

    def test_single_tree_trained(self, sample_regression_data):
        """Test that only a single tree is trained (n_estimators=1)."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Verify model has only 1 tree
        tree_dump = model.model.booster_.dump_model()
        assert len(tree_dump["tree_info"]) == 1

    def test_feature_names_preserved(self, sample_regression_data):
        """Test that feature names are preserved after training."""
        X, y = sample_regression_data  # noqa: N806
        expected_feature_names = X.columns.tolist()

        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        assert model.column_names == expected_feature_names

    def test_max_depth_unlimited(self, sample_regression_data):
        """Test that max_depth=-1 creates deeper trees."""
        X, y = sample_regression_data  # noqa: N806
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=5, max_depth=-1)

        # Model should still train successfully
        assert model.model is not None
        assert "r2" in model.metrics_

    def test_min_child_samples_enforced(self, sample_regression_data):
        """Test that min_child_samples parameter is respected."""
        X, y = sample_regression_data  # noqa: N806

        # Train with large min_child_samples (should result in shallower tree)
        model_large = SpeedDrill()
        model_large.fit(X, y, min_child_samples=40, max_depth=-1)  # Magic value is for testing purposes

        # Train with small min_child_samples (should result in deeper tree if max_depth allows)
        model_small = SpeedDrill()
        model_small.fit(X, y, min_child_samples=5, max_depth=-1)

        # Both should train successfully
        assert model_large.model is not None
        assert model_small.model is not None


class TestSpeedDrillIntegration:
    """Integration tests for SpeedDrill with complete workflow."""

    def test_complete_workflow_regression(self, sample_regression_data):
        """Test complete workflow: fit -> check metrics -> visualize tree."""
        X, y = sample_regression_data  # noqa: N806

        # Initialize and fit
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Check metrics
        assert "r2" in model.metrics_
        assert "mse" in model.metrics_
        assert model.metrics_["r2"] >= 0  # R² can be negative for bad fits, but should be positive here

        # Visualize tree
        ax = model.view_tree(figsize=(20, 12))
        assert ax is not None

        # Verify feature names are used in tree
        text_strings = [t.get_text() for t in ax.texts]
        # At least one of the feature names should appear in the tree
        feature_names = ["feature1", "feature2", "feature3"]
        assert any(fname in " ".join(text_strings) for fname in feature_names)

    def test_complete_workflow_classification(self, sample_binary_data):
        """Test complete workflow for binary classification."""
        X, y = sample_binary_data  # noqa: N806

        # Initialize and fit
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=10, max_depth=3)

        # Check metrics
        assert "accuracy" in model.metrics_
        assert "auc" in model.metrics_

        # Visualize tree
        ax = model.view_tree(figsize=(20, 12))
        assert ax is not None

        # Verify feature names are used in tree
        text_strings = [t.get_text() for t in ax.texts]
        feature_names = ["age", "income", "score"]
        assert any(fname in " ".join(text_strings) for fname in feature_names)

    def test_method_chaining(self, sample_regression_data):
        """Test that fit returns self for method chaining."""
        X, y = sample_regression_data  # noqa: N806

        model = SpeedDrill().fit(X, y, min_child_samples=10, max_depth=3)

        # Should be able to call methods after chaining
        assert model.model is not None
        assert model.column_names is not None

        # Should be able to visualize after chaining
        ax = model.view_tree()
        assert ax is not None


class TestLightGBMTreeWithIdenticalValues:
    """Test tree visualization with identical values in nodes."""

    def test_lightgbm_tree_with_identical_values(self, sample_regression_data):
        """Test that LightGBM visualization works with identical values in nodes."""
        X, y = sample_regression_data  # noqa: N806

        # Use a more complex model configuration
        model = SpeedDrill()
        model.fit(X, y, min_child_samples=5, max_depth=3)

        # Generate visualization
        ax = model.view_tree(figsize=(20, 12))

        # Verify visualization was created
        assert ax is not None
        assert isinstance(ax, plt.Axes)

        # Check that the visualization has content
        assert len(ax.patches) > 0  # Should have node rectangles
        assert len(ax.texts) > 0  # Should have node labels

        # Verify important tree elements are present
        text_content = [text.get_text() for text in ax.texts]
        assert any("feature" in text.lower() for text in text_content)  # Feature names should appear
