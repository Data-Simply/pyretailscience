"""Integration tests for Cross Shop Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.cross_shop import CrossShop


@pytest.mark.parametrize(
    "group_3_col",
    [
        "category_1_name",
        None,
    ],
)
def test_cross_shop_with_bigquery(transactions_table, group_3_col):
    """Test CrossShop with data fetched from BigQuery.

    This parameterized test verifies that CrossShop can be initialized
    and run with data from BigQuery using different combinations of group columns,
    value columns, and aggregation functions without throwing exceptions.
    """
    transactions_df = transactions_table.limit(5000)
    group_1_col = "brand_name"
    group_2_col = "category_0_name"
    group_1_vals = transactions_df[group_1_col].execute().dropna().unique()
    group_2_vals = transactions_df[group_2_col].execute().dropna().unique()

    group_1_val = group_1_vals[0]
    group_2_val = group_2_vals[0]

    group_3_val = None
    if group_3_col is not None:
        group_3_vals = transactions_df[group_3_col].execute().dropna().unique()
        if len(group_3_vals) == 0:
            pytest.skip(f"Not enough unique values for {group_3_col}")
        group_3_val = group_3_vals[0]

    labels = ["Group 1", "Group 2"] if group_3_col is None else ["Group 1", "Group 2", "Group 3"]

    CrossShop(
        df=transactions_table,
        group_1_col=group_1_col,
        group_1_val=group_1_val,
        group_2_col=group_2_col,
        group_2_val=group_2_val,
        group_3_col=group_3_col,
        group_3_val=group_3_val,
        labels=labels,
        value_col="unit_quantity",
        agg_func="count",
    )
