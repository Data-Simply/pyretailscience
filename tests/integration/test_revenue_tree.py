"""Unified integration tests for Revenue Tree Analysis with multiple database backends."""

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
def test_revenue_tree_integration(
    transactions_table,
    group_col,
    include_qty,
):
    """Integration test for RevenueTree using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        group_col: Group column parameter for analysis
        include_qty: Whether to include quantity in analysis
    """
    period_col = "transaction_date"
    limited_table = transactions_table.limit(10000)

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

    filtered_transactions = limited_table.select(columns_to_keep)

    revenue_tree = RevenueTree(
        df=filtered_transactions,
        period_col=period_col,
        p1_value="2023-05-24",
        p2_value="2023-04-15",
        group_col=group_col,
    )

    assert revenue_tree is not None
