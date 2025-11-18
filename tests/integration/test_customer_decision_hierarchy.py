"""Unified integration tests for Customer Decision Hierarchy Analysis with multiple database backends."""

import pytest

from pyretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy


@pytest.mark.parametrize(
    ("method", "exclude_same_transaction"),
    [
        ("yules_q", False),
        ("yules_q", None),
    ],
)
def test_customer_decision_hierarchy_integration(
    transactions_table,
    method,
    exclude_same_transaction,
):
    """Integration test for CustomerDecisionHierarchy using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        method: Method parameter for analysis
        exclude_same_transaction: Whether to exclude same transaction products
    """
    limited_table = transactions_table.limit(5000)
    transactions_df = limited_table.execute()

    customer_decision_hierarchy = CustomerDecisionHierarchy(
        df=transactions_df,
        product_col="product_name",
        exclude_same_transaction_products=exclude_same_transaction,
        method=method,
    )

    assert customer_decision_hierarchy is not None
