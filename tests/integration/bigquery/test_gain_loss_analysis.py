"""Integration tests for the GainLoss class using BigQuery data loaded into a DataFrame."""

import pytest

from pyretailscience.analysis.gain_loss import GainLoss


@pytest.mark.parametrize(
    ("focus_brand_id", "comparison_brand_id", "group_col", "value_col,agg_func"),
    [
        (1, 2, None, "unit_spend", "sum"),
        (3, 4, "category_1_name", "unit_spend", "sum"),
        (5, 6, "category_0_name", "unit_quantity", "sum"),
        (7, 8, "brand_name", "unit_cost", "mean"),
    ],
)
def test_gain_loss_with_bigquery(
    transactions_table,
    focus_brand_id,
    comparison_brand_id,
    group_col,
    value_col,
    agg_func,
):
    """Test GainLoss with data fetched from BigQuery.

    This parameterized test verifies that GainLoss can be initialized
    and run with data from BigQuery using different combinations of brands,
    grouping columns, value columns, and aggregation functions without throwing exceptions.
    """
    transactions_df = transactions_table.limit(5000).execute()

    min_date = transactions_df["transaction_date"].min()
    max_date = transactions_df["transaction_date"].max()
    mid_date = min_date + (max_date - min_date) / 2

    try:
        p1_index = transactions_df["transaction_date"] <= mid_date
        p2_index = transactions_df["transaction_date"] > mid_date

        available_brands = transactions_df["brand_id"].unique()
        actual_focus_brand = available_brands[0] if len(available_brands) > 0 else focus_brand_id
        actual_comparison_brand = available_brands[-1] if len(available_brands) > 1 else comparison_brand_id

        focus_group_index = transactions_df["brand_id"] == actual_focus_brand
        comparison_group_index = transactions_df["brand_id"] == actual_comparison_brand

        GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=f"Brand {actual_focus_brand}",
            comparison_group_index=comparison_group_index,
            comparison_group_name=f"Brand {actual_comparison_brand}",
            group_col=group_col,
            value_col=value_col,
            agg_func=agg_func,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"GainLoss failed with parameters focus_brand={focus_brand_id}, comparison_brand={comparison_brand_id}, group_col={group_col}, value_col={value_col}, agg_func={agg_func}: {e}",
        )
