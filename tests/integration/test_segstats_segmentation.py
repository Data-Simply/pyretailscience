"""Unified integration tests for segmentation statistics with multiple database backends."""

import pytest

from pyretailscience.segmentation.segstats import SegTransactionStats


@pytest.mark.parametrize(
    ("calc_total", "extra_aggs"),
    [
        (True, None),
        (True, {"unique_products": ("product_id", "nunique")}),
        (False, None),
        (False, {"unique_products": ("product_id", "nunique")}),
    ],
)
def test_seg_transaction_stats_integration(
    transactions_table,
    calc_total,
    extra_aggs,
):
    """Integration test for SegTransactionStats using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        calc_total: Whether to calculate total statistics
        extra_aggs: Extra aggregations to include
    """
    limited_table = transactions_table.limit(10000)

    seg_transaction_stats = SegTransactionStats(
        data=limited_table,
        segment_col=["category_0_name", "category_1_name"],
        calc_total=calc_total,
        extra_aggs=extra_aggs,
    )

    assert seg_transaction_stats is not None
    result = seg_transaction_stats.df
    assert result is not None


def test_seg_transaction_stats_with_unknown_customer_integration(transactions_table):
    """Integration test for SegTransactionStats with unknown_customer_value using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
    """
    limited_table = transactions_table.limit(10000)

    seg_transaction_stats = SegTransactionStats(
        data=limited_table,
        segment_col=["category_0_name", "category_1_name"],
        calc_total=True,
        unknown_customer_value=-1,
    )

    assert seg_transaction_stats is not None
    result = seg_transaction_stats.df
    assert result is not None

    assert "spend_unknown" in result.columns
    assert "transactions_unknown" in result.columns
    assert "spend_total" in result.columns
    assert "transactions_total" in result.columns
