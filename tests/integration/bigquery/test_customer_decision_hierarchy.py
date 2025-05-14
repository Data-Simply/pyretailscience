"""Integration tests for Customer Decision Hierarchy Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


@pytest.mark.parametrize(
    ("product_col", "method", "min_var_explained", "exclude_same_transaction"),
    [
        ("product_name", "truncated_svd", 0.8, True),
        ("category_1_name", "yules_q", None, True),
        ("brand_name", "truncated_svd", 0.7, True),
        ("category_0_name", "truncated_svd", 0.7, False),
    ],
)
def test_customer_decision_hierarchy_with_bigquery(
    transactions_table,
    product_col,
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
            product_col=product_col,
            exclude_same_transaction_products=exclude_same_transaction,
            method=method,
            min_var_explained=min_var_explained if min_var_explained is not None else 0.8,
        )
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"CustomerDecisionHierarchy failed with product_col={product_col}, method={method}: {e}")
