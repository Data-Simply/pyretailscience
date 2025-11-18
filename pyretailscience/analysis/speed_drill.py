"""SpeedDrill: Descriptive analytics tool using LightGBM decision trees.

SpeedDrill is a descriptive analytics tool for exploratory data analysis (EDA), not a predictive modeling tool.
It fits a single decision tree on the entire dataset to understand feature relationships and variance explanation.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

from pyretailscience.plots import bar, cohort


class SpeedDrill:
    """Descriptive analytics tool using single LightGBM decision tree for EDA.

    SpeedDrill fits a single decision tree on the full dataset to understand which features
    explain variance in the target variable and how the model segments/partitions the data.
    This is for exploratory data analysis, not prediction.

    Training metrics (R², MSE, accuracy, AUC) serve as data quality checks to verify features
    adequately explain variance, not as predictive performance measures.

    Attributes:
        model: Trained LightGBM model (LGBMRegressor or LGBMClassifier).
        column_names: List of feature names from training data.
        metrics_: Dict of training metrics for data quality assessment.
                 Binary: {"accuracy": float, "auc": float}
                 Regression: {"r2": float, "mse": float}

    Example:
        >>> from pyretailscience.analysis.speed_drill import SpeedDrill
        >>> model = SpeedDrill()
        >>> model.fit(X_train, y_train, min_child_samples=20, max_depth=5)
        >>> ax = model.view_tree(figsize=(20, 12))
        >>> print(model.metrics_)  # Data quality check
        >>> ax = model.plot_feature_importances(top_n=15)
    """

    model: lgb.LGBMModel | None
    column_names: list[str] | None
    metrics_: dict[str, float]

    def __init__(self) -> None:
        """Initialize SpeedDrill."""
        self.model = None
        self.column_names = None
        self.metrics_ = {}

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        min_child_samples: int,
        max_depth: int,
        **kwargs: Any,  # noqa: ANN401 - LightGBM params vary by model type
    ) -> "SpeedDrill":
        """Train a single LightGBM tree for descriptive analysis of the full dataset.

        Automatically detects target type and configures for binary classification or regression.
        For descriptive analytics, fits on entire dataset to understand feature relationships.

        Training metrics are stored in `self.metrics_` dict for data quality assessment:
        - Binary: {"accuracy": float, "auc": float}
        - Regression: {"r2": float, "mse": float}

        Args:
            x (pd.DataFrame): Input features DataFrame.
            y (pd.Series): Target variable Series (continuous or binary).
            min_child_samples (int): Minimum samples required in a leaf node.
            max_depth (int): Maximum tree depth (-1 for unlimited).
            **kwargs: Additional LightGBM parameters to override defaults.

        Returns:
            SpeedDrill: Self for method chaining.

        Raises:
            ValueError: If target type is not supported (must be continuous or binary).
        """
        # Store feature names
        self.column_names = x.columns.tolist()

        # Detect target type
        unique_values = y.nunique()
        binary_target_count = 2
        is_binary = unique_values == binary_target_count

        # Configure model based on target type
        if is_binary:
            # Binary classification
            default_params = {
                "n_estimators": 1,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "random_state": 42,
                "verbose": -1,
            }
            default_params.update(kwargs)
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            # Regression
            default_params = {
                "n_estimators": 1,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "random_state": 42,
                "verbose": -1,
            }
            default_params.update(kwargs)
            self.model = lgb.LGBMRegressor(**default_params)

        # Fit model on full dataset (descriptive analytics)
        self.model.fit(x, y)

        # Calculate metrics for data quality assessment
        y_pred = self.model.predict(x)

        if is_binary:
            # Binary classification metrics
            y_pred_proba = self.model.predict_proba(x)[:, 1]
            self.metrics_ = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "auc": float(roc_auc_score(y, y_pred_proba)),
            }
        else:
            # Regression metrics
            self.metrics_ = {
                "r2": float(r2_score(y, y_pred)),
                "mse": float(mean_squared_error(y, y_pred)),
            }

        return self

    def view_tree(self, figsize: tuple[float, float] = (20, 12)) -> Axes:
        """Visualize the trained decision tree using TreeGrid.

        Creates a pure Python visualization of the decision tree without external dependencies.
        Uses the TreeGrid system with LightGBMTreeNode for rendering.

        Args:
            figsize (tuple[float, float]): Figure size as (width, height) in inches. Defaults to (20, 12).

        Returns:
            matplotlib.axes.Axes: The axes object containing the tree visualization.

        Raises:
            ValueError: If model has not been trained yet.

        Example:
            >>> model = SpeedDrill()
            >>> model.fit(X, y, min_child_samples=20, max_depth=5)
            >>> ax = model.view_tree(figsize=(24, 16))
            >>> plt.savefig("tree.png", dpi=150, bbox_inches="tight")
        """
        from pyretailscience.plots.tree_diagram import LightGBMTreeNode, TreeGrid, lightgbm_tree_to_grid

        if not hasattr(self, "model") or self.model is None:
            msg = "Model has not been trained yet. Please call fit() first."
            raise ValueError(msg)

        if not hasattr(self.model, "booster_") or self.model.booster_ is None:
            msg = "Trained LightGBM model does not have a booster_ attribute. Cannot visualize tree."
            raise ValueError(msg)

        # Convert LightGBM tree to TreeGrid format
        tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(
            self.model.booster_,
            feature_names=self.column_names,
        )

        # Create TreeGrid with manual dimensions
        # Nodes are 3.5 wide x 1.7 tall, so spacing must be larger than node size
        grid = TreeGrid(
            tree_structure=tree_structure,
            num_rows=num_rows,
            num_cols=num_cols,
            node_class=LightGBMTreeNode,
            horizontal_spacing=4.5,  # Wide enough to prevent overlap (node width is 3.5)
            vertical_spacing=2.5,  # Tall enough for clear separation (node height is 1.7)
        )

        # Create figure with desired size and render tree
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax = grid.render(ax=ax)

        # Set axes limits to show the full tree
        plot_width = grid.col[grid.num_cols - 1] + grid.node_width
        plot_height = grid.row[0] + grid.node_height
        ax.set_xlim(0, plot_width)
        ax.set_ylim(0, plot_height)

        ax.axis("off")

        fig.suptitle("LightGBM Decision Tree", fontsize=16, fontweight="bold")

        return ax

    @staticmethod
    def generate_correlations_heatmap(
        x: pd.DataFrame,
        y: pd.Series,
        target_column: str,
        top_n: int = 60,
    ) -> Axes:
        """Generate a correlation heatmap for top features most correlated with target.

        Creates a Spearman correlation matrix heatmap showing relationships between
        the top N features (by absolute correlation with target) and the target variable.

        Args:
            x (pd.DataFrame): Input features DataFrame (numeric features only will be used).
            y (pd.Series): Target variable Series.
            target_column (str): Name to use for target column in correlation matrix.
            top_n (int): Number of top correlated features to include. Defaults to 60.

        Returns:
            matplotlib.axes.Axes: The axes object containing the heatmap.

        Example:
            >>> from pyretailscience.analysis.speed_drill import SpeedDrill
            >>> ax = SpeedDrill.generate_correlations_heatmap(
            ...     X, y, target_column="conversion_rate", top_n=20
            ... )
        """
        # Create dataframe with all numeric features + target for correlation analysis
        correlation_df = x.select_dtypes(include=[np.number]).copy()
        correlation_df[target_column] = y

        # Calculate correlation matrix
        correlation_matrix = correlation_df.corr(method="spearman")

        # Get correlations with target column, sort by absolute value
        abs_target_correlations = correlation_matrix[target_column].abs().sort_values(ascending=False)

        # Get top N features based on absolute correlation (excluding the target itself)
        top_abs_corr_features_base = abs_target_correlations.drop(
            labels=[target_column],
            errors="ignore",
        ).index.tolist()

        # Filter to include only features present in the correlation matrix
        top_abs_corr_features_filtered = [
            f for f in top_abs_corr_features_base if f in correlation_matrix.columns and f != target_column
        ]

        # Take the first top_n features, or fewer if not enough are available
        selected_abs_corr_features = top_abs_corr_features_filtered[:top_n]

        # Add the target column
        if target_column in correlation_matrix.columns:
            selected_abs_corr_features.append(target_column)

        # Create subset correlation matrix for top absolute-correlation features
        top_abs_correlation_matrix = correlation_matrix.loc[selected_abs_corr_features, selected_abs_corr_features]

        # Use PyRetailScience standard heatmap
        title = (
            f"Top {len(selected_abs_corr_features) - 1} Absolute Correlated Features Heatmap\n"
            "(Based on Absolute Correlation with Target)"
        )

        return cohort.plot(
            df=top_abs_correlation_matrix,
            cbar_label="Correlation",
            x_label="",
            y_label="",
            title=title,
            figsize=(30, 30),
            percentage=False,
        )

    def plot_feature_importances(
        self,
        top_n: int = 10,
        ax: Axes | None = None,
        title: str = "Feature Importances",
        **kwargs: Any,  # noqa: ANN401 - bar.plot kwargs vary
    ) -> Axes:
        """Generate a horizontal bar plot of the top N feature importances.

        Calculates feature importances based on 'gain' (Mean Decrease in Impurity)
        and plots them using PyRetailScience's standard bar plot.

        Args:
            top_n (int): Number of top features to display. Defaults to 10.
            ax (matplotlib.axes.Axes | None): Matplotlib Axes object to plot on. If None, creates new axes.
            title (str): Plot title. Defaults to "Feature Importances".
            **kwargs: Additional arguments passed to pyretailscience.plots.bar.plot()

        Returns:
            matplotlib.axes.Axes: Matplotlib Axes object with the plot.

        Raises:
            ValueError: If model has not been trained yet or all importances are zero.

        Example:
            >>> model = SpeedDrill()
            >>> model.fit(X, y, min_child_samples=20, max_depth=5)
            >>> ax = model.plot_feature_importances(top_n=15, title="Top 15 Features")
        """
        if not hasattr(self, "model") or self.model is None:
            msg = "Model has not been trained yet. Call fit() first."
            raise ValueError(msg)

        # Calculate feature importances
        importance = self.model.feature_importances_
        feature_names = self.column_names

        importance_df = pd.DataFrame(
            {
                "Feature_Name": feature_names,
                "Importance": importance,
            },
        ).sort_values(by="Importance", ascending=False)

        total_importance = importance_df["Importance"].sum()
        if total_importance > 0:
            importance_df["Importance_Percentage"] = (importance_df["Importance"] / total_importance) * 100
        else:
            importance_df["Importance_Percentage"] = 0

        # Select top N features
        plot_df = importance_df.head(top_n).copy()

        if plot_df.empty or plot_df["Importance_Percentage"].sum() == 0:
            msg = "No feature importances to plot (all importances are zero)."
            raise ValueError(msg)

        # Use PyRetailScience standard bar plot
        return bar.plot(
            df=plot_df,
            value_col="Importance_Percentage",
            x_col="Feature_Name",
            title=title,
            x_label="",
            y_label="Importance (%)",
            orientation="horizontal",
            ax=ax,
            **kwargs,
        )
