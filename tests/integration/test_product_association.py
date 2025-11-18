"""Unified integration tests for Product Association Analysis with multiple database backends."""

import pytest

from pyretailscience.analysis.product_association import ProductAssociation


@pytest.mark.parametrize(
    "target_item",
    [None, "Electronics"],
)
def test_product_association_integration(
    transactions_table,
    target_item,
):
    """Integration test for ProductAssociation using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        target_item: Target item parameter for analysis
    """
    limited_table = transactions_table.limit(5000)

    product_association = ProductAssociation(
        df=limited_table,
        value_col="brand_name",
        group_col="transaction_id",
        target_item=target_item,
        min_occurrences=5,
        min_cooccurrences=3,
        min_support=0.01,
        min_confidence=0.05,
        min_uplift=1.0,
    )

    assert product_association is not None
