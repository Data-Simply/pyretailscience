"""Shopping Mission Discovery using SPPMI and HDBSCAN.

This module implements shopping mission discovery using an NLP-inspired approach:
1. SPPMI (Shifted Positive Pointwise Mutual Information) for co-occurrence weighting
2. SVD for dimensionality reduction
3. HDBSCAN for density-based clustering

This approach is particularly good at:
- Handling popularity bias (common items don't dominate)
- Discovering clusters of varying density
- Identifying noise/outlier transactions
- Finding natural mission groupings without pre-specifying cluster count
"""

import hdbscan
import numpy as np
from scipy import sparse

from pyretailscience.analysis.shopping_missions import ShoppingMissionsBase
from pyretailscience.utils.embedding import compute_sppmi_embeddings


class ShoppingMissionsSPPMI(ShoppingMissionsBase):
    """Discover shopping missions using SPPMI + SVD + HDBSCAN.

    This implementation uses Shifted Positive Pointwise Mutual Information (SPPMI)
    to weight item co-occurrences, emphasizing meaningful associations while
    reducing popularity bias. HDBSCAN then identifies mission clusters of varying
    density.

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
        sppmi_shift (float): Shift constant for SPPMI (typically 1-5). Higher
            values create sparser embeddings. Defaults to 1.0.
        min_cluster_size (int): Minimum size of HDBSCAN clusters. Smaller values
            find more granular missions. Defaults to 5.
        min_samples (int | None): HDBSCAN min_samples parameter. If None, uses
            min_cluster_size. Defaults to None.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Attributes:
        sppmi_shift (float): SPPMI shift parameter used.
        min_cluster_size (int): HDBSCAN minimum cluster size.
        min_samples (int): HDBSCAN min_samples parameter.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'transaction_id': [1, 1, 2, 2, 3, 3],
        ...     'item': ['bread', 'milk', 'bread', 'butter', 'milk', 'eggs']
        ... })
        >>> missions = ShoppingMissionsSPPMI(
        ...     df,
        ...     transaction_col='transaction_id',
        ...     item_col='item',
        ...     min_cluster_size=2
        ... )
        >>> result = missions.df
        >>> summary = missions.get_mission_summary()
    """

    def __init__(
        self,
        df,
        transaction_col=None,
        item_col=None,
        min_items_per_transaction=2,
        n_components=50,
        sppmi_shift=1.0,
        min_cluster_size=5,
        min_samples=None,
        random_state=42,
    ) -> None:
        """Initialize ShoppingMissionsSPPMI.

        Args:
            df: Transaction data.
            transaction_col: Transaction ID column name.
            item_col: Item column name.
            min_items_per_transaction: Minimum items per transaction.
            n_components: SVD dimensions.
            sppmi_shift: SPPMI shift constant.
            min_cluster_size: Minimum HDBSCAN cluster size.
            min_samples: HDBSCAN min_samples (defaults to min_cluster_size).
            random_state: Random seed.
        """
        super().__init__(
            df=df,
            transaction_col=transaction_col,
            item_col=item_col,
            min_items_per_transaction=min_items_per_transaction,
            n_components=n_components,
            random_state=random_state,
        )

        self.sppmi_shift = sppmi_shift
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size

    def _compute_embeddings(self) -> sparse.csr_matrix:
        """Compute SPPMI embeddings from transaction-item matrix.

        Returns:
            sparse.csr_matrix: SPPMI-weighted item-item co-occurrence matrix.
        """
        # Compute SPPMI on the item-item co-occurrence
        # This creates an (n_items x n_items) matrix
        item_embeddings = compute_sppmi_embeddings(
            self._matrix,
            shift=self.sppmi_shift,
        )

        # Project transactions into this item embedding space
        # Result: (n_transactions x n_items) @ (n_items x n_items) = (n_transactions x n_items)
        transaction_embeddings = self._matrix @ item_embeddings

        return transaction_embeddings

    def _cluster_embeddings(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """Cluster reduced embeddings using HDBSCAN.

        Args:
            reduced_embeddings (np.ndarray): Reduced dimension embeddings.

        Returns:
            np.ndarray: Cluster labels for each transaction. -1 indicates noise.
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of Mass
        )

        labels = clusterer.fit_predict(reduced_embeddings)

        return labels
