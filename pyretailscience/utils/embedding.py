"""Embedding utilities for transaction and item vector representations.

This module provides shared utilities for converting transaction data into
embeddings suitable for clustering and similarity analysis. Supports multiple
weighting schemes (TF-IDF, SPPMI) and dimensionality reduction techniques.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def build_transaction_item_matrix(
    df: pd.DataFrame,
    transaction_col: str = "transaction_id",
    item_col: str = "item",
) -> tuple[sparse.csr_matrix, list[str], list[int]]:
    """Build a transaction-item matrix from transaction data.

    Creates a sparse binary matrix where rows represent transactions and
    columns represent items. Matrix[i,j] = 1 if transaction i contains item j.

    Args:
        df (pd.DataFrame): Transaction data with one row per transaction-item pair.
        transaction_col (str): Name of column containing transaction identifiers.
        item_col (str): Name of column containing item identifiers.

    Returns:
        tuple: Contains:
            - sparse.csr_matrix: Transaction-item binary matrix (n_transactions x n_items)
            - list[str]: Item names corresponding to matrix columns
            - list[int]: Transaction IDs corresponding to matrix rows

    Raises:
        ValueError: If required columns are missing from the DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'transaction_id': [1, 1, 2, 2, 3],
        ...     'item': ['milk', 'bread', 'milk', 'eggs', 'bread']
        ... })
        >>> matrix, items, txn_ids = build_transaction_item_matrix(df)
        >>> matrix.shape
        (3, 3)  # 3 transactions, 3 unique items
    """
    required_cols = [transaction_col, item_col]
    missing_cols = set(required_cols) - set(df.columns)
    if len(missing_cols) > 0:
        msg = f"The following columns are required but missing: {missing_cols}"
        raise ValueError(msg)

    # Get unique items and transactions
    unique_items = sorted(df[item_col].unique())
    unique_transactions = sorted(df[transaction_col].unique())

    # Create mappings
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    txn_to_idx = {txn: idx for idx, txn in enumerate(unique_transactions)}

    # Build sparse matrix
    row_indices = [txn_to_idx[txn] for txn in df[transaction_col]]
    col_indices = [item_to_idx[item] for item in df[item_col]]
    data = np.ones(len(df), dtype=np.int8)

    matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(unique_transactions), len(unique_items)),
        dtype=np.int8,
    )

    return matrix, unique_items, unique_transactions


def compute_tfidf_embeddings(
    transaction_item_matrix: sparse.csr_matrix,
    **kwargs: dict,
) -> sparse.csr_matrix:
    """Compute TF-IDF embeddings for transactions.

    Applies TF-IDF (Term Frequency-Inverse Document Frequency) transformation
    to emphasize items that are distinctive to specific transactions while
    down-weighting commonly purchased items.

    Args:
        transaction_item_matrix (sparse.csr_matrix): Binary transaction-item matrix.
        **kwargs: Additional arguments passed to sklearn's TfidfTransformer.

    Returns:
        np.ndarray: TF-IDF weighted matrix (same shape as input).

    Example:
        >>> matrix = sparse.csr_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        >>> tfidf_matrix = compute_tfidf_embeddings(matrix)
        >>> tfidf_matrix.shape
        (3, 3)
    """
    from sklearn.feature_extraction.text import TfidfTransformer

    transformer = TfidfTransformer(**kwargs)
    return transformer.fit_transform(transaction_item_matrix)


def compute_sppmi_embeddings(
    transaction_item_matrix: sparse.csr_matrix,
    shift: float = 1.0,
) -> sparse.csr_matrix:
    """Compute SPPMI (Shifted Positive Pointwise Mutual Information) embeddings.

    SPPMI is a weighting scheme that emphasizes meaningful co-occurrences while
    reducing the bias toward popular items. It's based on PMI from NLP, which
    measures how much more likely two items co-occur than expected by chance.

    PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    SPPMI(i,j) = max(PMI(i,j) - shift, 0)

    The shift parameter controls sparsity and reduces noise.

    Args:
        transaction_item_matrix (sparse.csr_matrix): Binary transaction-item matrix
            (n_transactions x n_items).
        shift (float): Shift constant (typically 1-5). Higher values create sparser
            embeddings by requiring stronger associations. Defaults to 1.0.

    Returns:
        sparse.csr_matrix: SPPMI-weighted item-item co-occurrence matrix
            (n_items x n_items).

    Example:
        >>> matrix = sparse.csr_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        >>> sppmi_matrix = compute_sppmi_embeddings(matrix, shift=1.0)
        >>> sppmi_matrix.shape
        (3, 3)  # item x item co-occurrence
    """
    # Ensure binary matrix
    transaction_item_matrix = transaction_item_matrix.astype(bool).astype(float)

    # Compute item-item co-occurrence matrix
    # C[i,j] = number of transactions containing both items i and j
    cooccurrence = transaction_item_matrix.T @ transaction_item_matrix

    # Total number of transactions
    n_transactions = transaction_item_matrix.shape[0]

    # Item frequencies (how many transactions contain each item)
    item_counts = np.array(transaction_item_matrix.sum(axis=0)).flatten()

    # Compute probabilities
    # P(i,j) = cooccurrence[i,j] / n_transactions
    p_ij = cooccurrence / n_transactions

    # P(i) * P(j) for all pairs
    # Using outer product to create matrix of expected probabilities
    p_i = item_counts / n_transactions
    p_i_p_j = np.outer(p_i, p_i)

    # Compute PMI = log(P(i,j) / (P(i) * P(j)))
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((p_ij.toarray() + epsilon) / (p_i_p_j + epsilon))
        # Replace inf and -inf with 0
        pmi[~np.isfinite(pmi)] = 0

    # Apply shift and positive threshold: SPPMI = max(PMI - shift, 0)
    sppmi = np.maximum(pmi - shift, 0)

    # Convert back to sparse matrix
    return sparse.csr_matrix(sppmi)


def reduce_dimensions(
    embeddings: sparse.csr_matrix | np.ndarray,
    n_components: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce dimensionality of embeddings using Truncated SVD.

    Applies SVD to create a lower-dimensional representation that captures
    the most important patterns in the data. Useful for reducing noise and
    computational complexity before clustering.

    Args:
        embeddings (sparse.csr_matrix | np.ndarray): High-dimensional embeddings
            (n_samples x n_features).
        n_components (int): Number of dimensions to keep. Should be less than
            min(n_samples, n_features). Defaults to 50.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: Reduced embeddings (n_samples x n_components).

    Raises:
        ValueError: If n_components is larger than available dimensions.

    Example:
        >>> embeddings = sparse.random(100, 500, density=0.1)
        >>> reduced = reduce_dimensions(embeddings, n_components=50)
        >>> reduced.shape
        (100, 50)
    """
    n_samples, n_features = embeddings.shape
    max_components = min(n_samples, n_features) - 1

    if n_components >= max_components:
        msg = f"n_components ({n_components}) must be less than min(n_samples, n_features) - 1 ({max_components})"
        raise ValueError(msg)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(embeddings)


def filter_min_items_per_transaction(
    df: pd.DataFrame,
    transaction_col: str = "transaction_id",
    min_items: int = 2,
) -> pd.DataFrame:
    """Filter out transactions with fewer than minimum items.

    Single-item transactions don't provide co-occurrence information,
    so they're typically excluded from mission discovery.

    Args:
        df (pd.DataFrame): Transaction data with one row per transaction-item pair.
        transaction_col (str): Name of column containing transaction identifiers.
        min_items (int): Minimum number of items required per transaction.
            Defaults to 2.

    Returns:
        pd.DataFrame: Filtered DataFrame with only transactions meeting the
            minimum item requirement.

    Example:
        >>> df = pd.DataFrame({
        ...     'transaction_id': [1, 1, 2, 3, 3, 3],
        ...     'item': ['a', 'b', 'c', 'd', 'e', 'f']
        ... })
        >>> filtered = filter_min_items_per_transaction(df, min_items=2)
        >>> filtered['transaction_id'].unique().tolist()
        [1, 3]  # Transaction 2 was removed (only 1 item)
    """
    item_counts = df.groupby(transaction_col).size()
    valid_transactions = item_counts[item_counts >= min_items].index
    filtered_df = df[df[transaction_col].isin(valid_transactions)]

    return filtered_df.reset_index(drop=True)
