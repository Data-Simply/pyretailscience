"""Integration tests for Revenue Tree Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.revenue_tree import RevenueTree
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


@pytest.mark.parametrize(
    ("group_col", "include_qty"),
    [
        (None, True),
        (None, False),
        ("category_0_name", True),
        ("category_0_name", False),
    ],
)
def test_revenue_tree_with_bigquery(
    transactions_table,
    group_col,
    include_qty,
):
    """Test RevenueTree with data fetched from BigQuery.

    This parameterized test verifies that RevenueTree can be initialized
    and process data from BigQuery using different group columns
    without throwing exceptions.
    """
    period_col = "transaction_date"

    limited_transactions = transactions_table.limit(10000)
    columns_to_keep = [
        cols.customer_id,
        cols.transaction_id,
        cols.unit_spend,
        period_col,
    ]

    if include_qty:
        columns_to_keep.append(cols.unit_qty)
    if group_col:
        columns_to_keep.append(group_col)

    filtered_transactions = limited_transactions.select(columns_to_keep)

    RevenueTree(
        df=filtered_transactions,
        period_col=period_col,
        p1_value="2023-05-24",
        p2_value="2023-04-15",
        group_col=group_col,
    )
