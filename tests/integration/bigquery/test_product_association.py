"""Integration tests for Product Association Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.product_association import ProductAssociation


@pytest.mark.parametrize(
    ("value_col", "group_col", "target_item"),
    [
        ("product_name", "customer_id", None),
        ("product_id", "customer_id", None),
        ("brand_name", "transaction_id", None),
        ("category_1_name", "customer_id", "Electronics"),
    ],
)
def test_product_association_with_bigquery(
    transactions_table,
    value_col,
    group_col,
    target_item,
):
    """Test ProductAssociation with data fetched from BigQuery.

    This parameterized test verifies that ProductAssociation can be initialized
    and process data from BigQuery using different combinations of value columns,
    group columns, and target items without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(5000)

    try:
        ProductAssociation(
            df=limited_transactions,
            value_col=value_col,
            group_col=group_col,
            target_item=target_item,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"ProductAssociation failed with value_col={value_col}, group_col={group_col}, target_item={target_item}: {e}",
        )


@pytest.mark.parametrize(
    ("min_occurrences", "min_cooccurrences", "min_support", "min_confidence", "min_uplift"),
    [
        (1, 1, 0.0, 0.0, 0.0),
        (5, 3, 0.01, 0.05, 1.0),
        (10, 5, 0.02, 0.1, 1.5),
    ],
)
def test_product_association_filtering_with_bigquery(
    transactions_table,
    min_occurrences,
    min_cooccurrences,
    min_support,
    min_confidence,
    min_uplift,
):
    """Test ProductAssociation filtering parameters with BigQuery data.

    This parameterized test verifies that ProductAssociation can process BigQuery data
    with different filtering parameters without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(3000)

    try:
        ProductAssociation(
            df=limited_transactions,
            value_col="product_name",
            group_col="customer_id",
            min_occurrences=min_occurrences,
            min_cooccurrences=min_cooccurrences,
            min_support=min_support,
            min_confidence=min_confidence,
            min_uplift=min_uplift,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"ProductAssociation filtering test failed: {e}")
