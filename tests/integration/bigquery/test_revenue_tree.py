"""Integration tests for Revenue Tree Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


@pytest.mark.parametrize(
    ("period_col", "group_col"),
    [
        ("transaction_date", None),
        ("transaction_date", "category_0_name"),
        ("transaction_date", "brand_name"),
        ("transaction_date", "store_id"),
    ],
)
def test_revenue_tree_with_bigquery(
    transactions_table,
    period_col,
    group_col,
):
    """Test RevenueTree with data fetched from BigQuery.

    This parameterized test verifies that RevenueTree can be initialized
    and process data from BigQuery using different period and group columns
    without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(10000)

    try:
        period_values = limited_transactions.select(period_col).distinct().limit(2).execute()

        p1_value = period_values[period_col].iloc[0]
        p2_value = period_values[period_col].iloc[1]

        RevenueTree(
            df=limited_transactions,
            period_col=period_col,
            p1_value=p1_value,
            p2_value=p2_value,
            group_col=group_col,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"RevenueTree failed with period_col={period_col}, group_col={group_col}: {e}")


def test_calc_tree_kpis_with_bigquery(transactions_table):
    """Test calc_tree_kpis function with data from BigQuery.

    This test verifies that the calc_tree_kpis function can process data derived
    from BigQuery without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(8000)

    try:
        period_values = limited_transactions.select("transaction_date").distinct().limit(2).execute()

        p1_value = period_values["transaction_date"].iloc[0]
        p2_value = period_values["transaction_date"].iloc[1]

        df, p1_index, p2_index = RevenueTree._agg_data(
            df=limited_transactions,
            period_col="transaction_date",
            p1_value=p1_value,
            p2_value=p2_value,
            group_col=None,
        )

        calc_tree_kpis(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"calc_tree_kpis failed: {e}")


@pytest.mark.parametrize(
    ("period_col", "include_qty"),
    [
        ("transaction_date", True),
        ("transaction_date", False),
    ],
)
def test_revenue_tree_quantity_handling_with_bigquery(
    transactions_table,
    period_col,
    include_qty,
):
    """Test RevenueTree with and without quantity columns using BigQuery data.

    This test verifies that RevenueTree can process BigQuery data both with and
    without quantity-related columns without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(6000)

    try:
        period_values = limited_transactions.select(period_col).distinct().limit(2).execute()

        p1_value = period_values[period_col].iloc[0]
        p2_value = period_values[period_col].iloc[1]

        if include_qty:
            columns_to_keep = [
                cols.customer_id,
                cols.transaction_id,
                cols.unit_spend,
                cols.unit_qty,
                period_col,
            ]
        else:
            columns_to_keep = [
                cols.customer_id,
                cols.transaction_id,
                cols.unit_spend,
                period_col,
            ]

        filtered_transactions = limited_transactions.select(columns_to_keep)

        RevenueTree(
            df=filtered_transactions,
            period_col=period_col,
            p1_value=p1_value,
            p2_value=p2_value,
            group_col=None,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"RevenueTree quantity handling test failed with include_qty={include_qty}: {e}")
