"""Shopping Mission Discovery using TF-IDF and KMeans.

This module implements shopping mission discovery using a standard ML approach:
1. TF-IDF (Term Frequency-Inverse Document Frequency) for item weighting
2. SVD for dimensionality reduction
3. KMeans for clustering

This approach is particularly good at:
- Fast computation with large datasets
- Reproducible results (deterministic with seed)
- Simple interpretation and tuning
- Works well when number of missions is known or estimated
"""

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from pyretailscience.analysis.shopping_missions import ShoppingMissionsBase
from pyretailscience.utils.embedding import compute_tfidf_embeddings


class ShoppingMissionsTFIDF(ShoppingMissionsBase):
    """Discover shopping missions using TF-IDF + SVD + KMeans.

    This implementation uses TF-IDF (Term Frequency-Inverse Document Frequency)
    to weight items in transactions, emphasizing distinctive purchases while
    down-weighting common items. KMeans then identifies a fixed number of
    mission clusters.

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
        n_missions (int): Number of mission clusters to discover. Defaults to 5.
        max_iter (int): Maximum iterations for KMeans. Defaults to 300.
        n_init (int): Number of KMeans initializations. Defaults to 10.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Attributes:
        n_missions (int): Number of clusters to find.
        max_iter (int): Maximum KMeans iterations.
        n_init (int): Number of KMeans runs.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'transaction_id': [1, 1, 2, 2, 3, 3],
        ...     'item': ['bread', 'milk', 'bread', 'butter', 'milk', 'eggs']
        ... })
        >>> missions = ShoppingMissionsTFIDF(
        ...     df,
        ...     transaction_col='transaction_id',
        ...     item_col='item',
        ...     n_missions=2
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
        n_missions=5,
        max_iter=300,
        n_init=10,
        random_state=42,
    ) -> None:
        """Initialize ShoppingMissionsTFIDF.

        Args:
            df: Transaction data.
            transaction_col: Transaction ID column name.
            item_col: Item column name.
            min_items_per_transaction: Minimum items per transaction.
            n_components: SVD dimensions.
            n_missions: Number of mission clusters.
            max_iter: Maximum KMeans iterations.
            n_init: Number of KMeans initializations.
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

        self.n_missions = n_missions
        self.max_iter = max_iter
        self.n_init = n_init

    def _compute_embeddings(self) -> sparse.csr_matrix:
        """Compute TF-IDF embeddings from transaction-item matrix.

        Returns:
            sparse.csr_matrix: TF-IDF weighted transaction-item matrix.
        """
        # Apply TF-IDF transformation
        # This creates an (n_transactions x n_items) matrix
        tfidf_embeddings = compute_tfidf_embeddings(self._matrix)

        return tfidf_embeddings

    def _cluster_embeddings(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """Cluster reduced embeddings using KMeans.

        Args:
            reduced_embeddings (np.ndarray): Reduced dimension embeddings.

        Returns:
            np.ndarray: Cluster labels for each transaction (0 to n_missions-1).
        """
        # Ensure n_missions doesn't exceed number of samples
        n_samples = reduced_embeddings.shape[0]
        n_clusters = min(self.n_missions, n_samples)

        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )

        labels = kmeans.fit_predict(reduced_embeddings)

        return labels
