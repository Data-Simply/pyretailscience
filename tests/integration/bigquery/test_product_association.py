"""Integration tests for Product Association Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.product_association import ProductAssociation


@pytest.mark.parametrize(
    "target_item",
    [None, "Electronics"],
)
def test_product_association_with_bigquery(
    transactions_table,
    target_item,
):
    """Test ProductAssociation with data fetched from BigQuery.

    This parameterized test verifies that ProductAssociation can be initialized
    and process data from BigQuery using target items without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(5000)

    ProductAssociation(
        df=limited_transactions,
        value_col="brand_name",
        group_col="transaction_id",
        target_item=target_item,
        min_occurrences=5,
        min_cooccurrences=3,
        min_support=0.01,
        min_confidence=0.05,
        min_uplift=1.0,
    )
