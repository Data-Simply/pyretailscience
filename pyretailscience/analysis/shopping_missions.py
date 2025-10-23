"""Shopping Mission Discovery from Transaction Baskets.

## Business Context

Shopping missions represent the underlying purpose behind a customer's store visit.
Rather than analyzing individual products, mission discovery reveals the "why" behind
shopping trips - whether customers are doing a quick lunch grab, weekly grocery stock-up,
party preparation, or evening snack run. Understanding these missions transforms how
retailers approach merchandising, store layout, promotions, and inventory management.

## The Business Problem

Retailers struggle to optimize store experiences when they only see individual product
purchases without understanding the broader shopping context. Without mission insights:
- Store layouts don't support how customers actually shop
- Promotions miss cross-category opportunities aligned with mission needs
- Inventory planning doesn't account for mission-driven demand spikes
- Personalization efforts focus on products instead of shopping occasions
- New store formats are designed without understanding shopping patterns

## Real-World Applications

1. **Strategic Store Layout**
   - Group products by mission rather than traditional category
   - Create "Quick Lunch" zones with sandwiches, drinks, and snacks together
   - Design "Breakfast Mission" endcaps combining cereal, milk, and fruit
   - Position impulse items relevant to detected missions

2. **Mission-Based Promotions**
   - Bundle promotions around mission needs (e.g., "Game Day Bundle")
   - Target customers based on their historical mission patterns
   - Time promotions to match mission peaks (breakfast rush, dinner prep)
   - Cross-category discounts aligned with mission item sets

3. **Inventory & Assortment**
   - Stock mission-critical items together to prevent stockouts
   - Optimize shelf space based on mission popularity
   - Introduce new products that fit existing mission patterns
   - Reduce slow-moving SKUs that don't fit any mission

4. **Personalized Shopping**
   - Recommend items that complete a customer's current mission
   - Show mission-relevant products rather than just similar items
   - Alert customers to mission deals when they're in-store
   - Predict next visit mission based on history

5. **Store Format Design**
   - Design convenience stores around most common missions
   - Create express checkout for single-mission trips
   - Develop micro-fulfillment strategies by mission
   - Test new formats focused on specific mission types

## Implementation Approach

This module discovers missions by:
1. Converting transactions into item co-occurrence patterns
2. Creating embeddings that capture shopping context
3. Clustering similar shopping trips into mission groups
4. Profiling missions to understand their characteristics

Two implementation approaches are available:
- **SPPMI**: Advanced NLP-inspired weighting for co-occurrence
- **TF-IDF**: Standard information retrieval approach

Both use dimensionality reduction (SVD) and clustering (KMeans/HDBSCAN)
to identify natural mission groupings.

## Base Class

This module provides a base class that can be extended with different
embedding and clustering strategies. See:
- `shopping_missions_sppmi.py` for SPPMI + HDBSCAN implementation
- `shopping_missions_tfidf.py` for TF-IDF + KMeans implementation
"""

import ibis
import numpy as np
import pandas as pd
from scipy import sparse

from pyretailscience.options import get_option
from pyretailscience.utils.embedding import (
    build_transaction_item_matrix,
    filter_min_items_per_transaction,
    reduce_dimensions,
)


class ShoppingMissionsBase:
    """Base class for shopping mission discovery implementations.

    This class provides common functionality for mission discovery that can be
    extended with different embedding and clustering strategies.

    Args:
        df (pd.DataFrame | ibis.Table): Transaction data with one row per
            transaction-item pair.
        transaction_col (str | None): Column name for transaction identifiers.
            Defaults to option column.customer_id.
        item_col (str | None): Column name for item identifiers.
            Defaults to option column.product.
        min_items_per_transaction (int): Minimum items required per transaction.
            Transactions with fewer items are filtered out. Defaults to 2.
        n_components (int): Number of dimensions for SVD reduction. Should be
            less than the number of unique items. Defaults to 50.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Attributes:
        transaction_col (str): Name of transaction identifier column.
        item_col (str): Name of item identifier column.
        min_items_per_transaction (int): Minimum items filter threshold.
        n_components (int): SVD dimension count.
        random_state (int): Random seed.

    Note:
        This is a base class. Use specific implementations like
        ShoppingMissionsSPPMI or ShoppingMissionsTFIDF instead.
    """

    _cached_df: pd.DataFrame | None = None
    _mission_profiles: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        transaction_col: str | None = None,
        item_col: str | None = None,
        min_items_per_transaction: int = 2,
        n_components: int = 50,
        random_state: int = 42,
    ) -> None:
        """Initialize ShoppingMissionsBase.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data.
            transaction_col (str | None): Transaction ID column name.
            item_col (str | None): Item column name.
            min_items_per_transaction (int): Minimum items per transaction.
            n_components (int): SVD dimensions.
            random_state (int): Random seed.

        Raises:
            ValueError: If required columns are missing.
            ValueError: If min_items_per_transaction < 2.
            ValueError: If n_components < 2.
        """
        # Set defaults from options
        self.transaction_col = transaction_col or get_option("column.customer_id")
        self.item_col = item_col or get_option("column.product")

        # Validate parameters
        min_items_threshold = 2
        if min_items_per_transaction < min_items_threshold:
            msg = f"min_items_per_transaction must be at least {min_items_threshold}"
            raise ValueError(msg)

        min_components_threshold = 2
        if n_components < min_components_threshold:
            msg = f"n_components must be at least {min_components_threshold}"
            raise ValueError(msg)

        # Convert ibis to pandas if needed
        if isinstance(df, ibis.Table):
            df = df.execute()

        # Validate required columns
        required_cols = [self.transaction_col, self.item_col]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        # Filter transactions by minimum items
        df_filtered = filter_min_items_per_transaction(
            df,
            transaction_col=self.transaction_col,
            min_items=min_items_per_transaction,
        )

        if len(df_filtered) == 0:
            msg = f"No transactions remain after filtering for {min_items_per_transaction}+ items"
            raise ValueError(msg)

        # Store parameters
        self.min_items_per_transaction = min_items_per_transaction
        self.n_components = n_components
        self.random_state = random_state

        # Store filtered data
        self._df_filtered = df_filtered

        # Build transaction-item matrix
        self._matrix, self._items, self._transaction_ids = build_transaction_item_matrix(
            df_filtered,
            transaction_col=self.transaction_col,
            item_col=self.item_col,
        )

    def _compute_embeddings(self) -> np.ndarray | sparse.csr_matrix:
        """Compute embeddings from transaction-item matrix.

        This method should be implemented by subclasses to apply their
        specific embedding strategy (TF-IDF, SPPMI, etc.).

        Returns:
            np.ndarray | sparse.csr_matrix: Embeddings matrix.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        msg = "Subclasses must implement _compute_embeddings"
        raise NotImplementedError(msg)

    def _cluster_embeddings(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """Cluster reduced embeddings into missions.

        This method should be implemented by subclasses to apply their
        specific clustering strategy (KMeans, HDBSCAN, etc.).

        Args:
            reduced_embeddings (np.ndarray): Reduced dimension embeddings.

        Returns:
            np.ndarray: Cluster labels for each transaction.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        msg = "Subclasses must implement _cluster_embeddings"
        raise NotImplementedError(msg)

    @property
    def df(self) -> pd.DataFrame:
        """Get transactions with assigned mission labels.

        Returns:
            pd.DataFrame: Original transactions with added 'mission' column
                containing cluster assignments.
        """
        if self._cached_df is None:
            # Compute embeddings
            embeddings = self._compute_embeddings()

            # Reduce dimensions
            reduced_embeddings = reduce_dimensions(
                embeddings,
                n_components=min(self.n_components, min(embeddings.shape) - 1),
                random_state=self.random_state,
            )

            # Cluster
            mission_labels = self._cluster_embeddings(reduced_embeddings)

            # Create DataFrame with transaction IDs and missions
            result_df = pd.DataFrame(
                {
                    self.transaction_col: self._transaction_ids,
                    "mission": mission_labels,
                },
            )

            # Merge with original filtered data to get items
            self._cached_df = self._df_filtered.merge(result_df, on=self.transaction_col)

        return self._cached_df

    def get_mission_profile(self, mission_id: int, top_n: int = 10) -> pd.DataFrame:
        """Get top items and statistics for a specific mission.

        Args:
            mission_id (int): The mission cluster ID to profile.
            top_n (int): Number of top items to return. Defaults to 10.

        Returns:
            pd.DataFrame: Top items for the mission with columns:
                - item: Item name
                - frequency: Number of transactions containing this item
                - percentage: Percentage of mission transactions with this item

        Raises:
            ValueError: If mission_id is not found in the data.
        """
        mission_df = self.df[self.df["mission"] == mission_id]

        if len(mission_df) == 0:
            msg = f"Mission {mission_id} not found in results"
            raise ValueError(msg)

        # Count item frequencies in this mission
        item_counts = mission_df.groupby(self.item_col).size().reset_index(name="frequency")

        # Calculate percentage of mission transactions containing each item
        n_mission_transactions = mission_df[self.transaction_col].nunique()
        item_counts["percentage"] = (item_counts["frequency"] / n_mission_transactions) * 100

        # Sort by frequency and get top N
        top_items = item_counts.sort_values("frequency", ascending=False).head(top_n)

        return top_items.reset_index(drop=True)

    def get_mission_summary(self) -> pd.DataFrame:
        """Get summary statistics for all discovered missions.

        Returns:
            pd.DataFrame: Summary with columns:
                - mission: Mission ID
                - n_transactions: Number of transactions in mission
                - avg_items_per_transaction: Average basket size
                - unique_items: Number of unique items in mission
                - top_item: Most frequent item in mission
        """
        mission_stats = []

        for mission_id in sorted(self.df["mission"].unique()):
            mission_df = self.df[self.df["mission"] == mission_id]

            # Basic stats
            n_transactions = mission_df[self.transaction_col].nunique()
            avg_items = mission_df.groupby(self.transaction_col).size().mean()
            unique_items = mission_df[self.item_col].nunique()

            # Top item
            top_item = mission_df.groupby(self.item_col).size().idxmax()

            mission_stats.append(
                {
                    "mission": mission_id,
                    "n_transactions": n_transactions,
                    "avg_items_per_transaction": round(avg_items, 2),
                    "unique_items": unique_items,
                    "top_item": top_item,
                },
            )

        return pd.DataFrame(mission_stats)
