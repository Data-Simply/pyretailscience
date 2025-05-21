"""Integration tests for segmentation statistics using BigQuery data."""

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
def test_seg_transaction_stats_with_bigquery(
    transactions_table,
    calc_total,
    extra_aggs,
):
    """Test SegTransactionStats with data fetched from BigQuery.

    This test verifies that SegTransactionStats can process data directly from
    a BigQuery connection using Ibis without throwing exceptions.
    """
    limited_table = transactions_table.limit(10000)

    result = SegTransactionStats(
        data=limited_table,
        segment_col=["category_0_name", "category_1_name"],
        calc_total=calc_total,
        extra_aggs=extra_aggs,
    )
    assert result is not None

    df = result.df
    assert df is not None
