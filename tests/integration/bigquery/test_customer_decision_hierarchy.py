"""Integration tests for Customer Decision Hierarchy Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy


@pytest.mark.parametrize(
    ("method", "exclude_same_transaction"),
    [
        ("truncated_svd", False),
        ("truncated_svd", None),
        ("yules_q", False),
        ("yules_q", None),
    ],
)
def test_customer_decision_hierarchy_with_bigquery(
    transactions_table,
    method,
    exclude_same_transaction,
):
    """Test CustomerDecisionHierarchy with data fetched from BigQuery.

    This parameterized test verifies that CustomerDecisionHierarchy can be initialized
    and run with data from BigQuery using different combinations of product columns
    and methods without throwing exceptions.
    """
    transactions_df = transactions_table.limit(5000).execute()

    CustomerDecisionHierarchy(
        df=transactions_df,
        product_col="product_name",
        exclude_same_transaction_products=exclude_same_transaction,
        method=method,
    )
