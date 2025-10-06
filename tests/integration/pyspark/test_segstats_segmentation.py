"""Integration tests for segmentation statistics using PySpark data."""

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
def test_seg_transaction_stats_with_pyspark(
    transactions_table,
    calc_total,
    extra_aggs,
):
    """Test SegTransactionStats with data fetched from PySpark.

    This test verifies that SegTransactionStats can process data directly from
    a PySpark connection using Ibis without throwing exceptions.
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


def test_seg_transaction_stats_with_unknown_customer_pyspark(transactions_table):
    """Test SegTransactionStats with unknown_customer_value using PySpark.

    This test verifies that the unknown customer tracking feature works correctly
    with PySpark backend.
    """
    limited_table = transactions_table.limit(10000)

    result = SegTransactionStats(
        data=limited_table,
        segment_col=["category_0_name", "category_1_name"],
        calc_total=True,
        unknown_customer_value=-1,
    )
    assert result is not None

    df = result.df
    assert df is not None

    # Verify unknown customer columns exist
    assert "spend_unknown" in df.columns
    assert "transactions_unknown" in df.columns
    assert "spend_total" in df.columns
    assert "transactions_total" in df.columns
