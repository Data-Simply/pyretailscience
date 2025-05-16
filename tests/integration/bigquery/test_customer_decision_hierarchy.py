"""Integration tests for Customer Decision Hierarchy Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


@pytest.mark.parametrize(
    ("method", "min_var_explained", "exclude_same_transaction"),
    [
        ("truncated_svd", 0.7, False),
        ("truncated_svd", 0.7, None),
        ("truncated_svd", None, False),
        ("yules_q", 0.7, False),
        ("yules_q", 0.7, None),
        ("yules_q", None, False),
        ("yules_q", None, None),
    ],
)
def test_customer_decision_hierarchy_with_bigquery(
    transactions_table,
    method,
    min_var_explained,
    exclude_same_transaction,
):
    """Test CustomerDecisionHierarchy with data fetched from BigQuery.

    This parameterized test verifies that CustomerDecisionHierarchy can be initialized
    and run with data from BigQuery using different combinations of product columns
    and methods without throwing exceptions.
    """
    transactions_df = transactions_table.limit(5000).execute()

    try:
        CustomerDecisionHierarchy(
            df=transactions_df,
            product_col="product_name",
            exclude_same_transaction_products=exclude_same_transaction,
            method=method,
            min_var_explained=min_var_explained if min_var_explained is not None else 0.8,
        )
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"CustomerDecisionHierarchy failed with, method={method}: {e}")
