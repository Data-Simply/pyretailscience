"""Integration tests for segmentation statistics using BigQuery data."""

import pytest

from pyretailscience.segmentation.segstats import SegTransactionStats


@pytest.mark.parametrize(
    ("segment_col", "calc_total", "extra_aggs"),
    [
        # Test with single segment column
        ("brand_name", True, None),
        # Test with multiple segment columns
        (["category_0_name", "category_1_name"], True, None),
        # Test with extra aggregations
        ("store_id", False, {"unique_products": ("product_id", "nunique")}),
        # Test with category segmentation and stores aggregation
        ("category_1_name", True, {"store_count": ("store_id", "nunique")}),
    ],
)
def test_seg_transaction_stats_with_bigquery(
    transactions_table,
    segment_col,
    calc_total,
    extra_aggs,
):
    """Test SegTransactionStats with data fetched from BigQuery.

    This test verifies that SegTransactionStats can process data directly from
    a BigQuery connection using Ibis without throwing exceptions.
    """
    try:
        limited_table = transactions_table.limit(10000)

        SegTransactionStats(
            data=limited_table,
            segment_col=segment_col,
            calc_total=calc_total,
            extra_aggs=extra_aggs,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"SegTransactionStats failed with segment_col={segment_col}, calc_total={calc_total}, extra_aggs={extra_aggs}: {e}",
        )
