"""Unified integration tests for Cross Shop Analysis with multiple database backends."""

import pytest

from pyretailscience.analysis.cross_shop import CrossShop


@pytest.mark.parametrize(
    "group_3_col",
    [
        "category_1_name",
        None,
    ],
)
def test_cross_shop_integration(transactions_table, group_3_col):
    """Integration test for CrossShop using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        group_3_col: Third grouping column parameter for testing
    """
    limited_table = transactions_table.limit(5000)
    group_1_col = "brand_name"
    group_2_col = "category_0_name"
    group_1_vals = limited_table[group_1_col].execute().dropna().unique()
    group_2_vals = limited_table[group_2_col].execute().dropna().unique()

    group_1_val = group_1_vals[0]
    group_2_val = group_2_vals[0]

    group_3_val = None
    if group_3_col is not None:
        group_3_vals = limited_table[group_3_col].execute().dropna().unique()
        if len(group_3_vals) == 0:
            pytest.skip(f"Not enough unique values for {group_3_col}")
        group_3_val = group_3_vals[0]

    labels = ["Group 1", "Group 2"] if group_3_col is None else ["Group 1", "Group 2", "Group 3"]

    cross_shop = CrossShop(
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

    assert cross_shop is not None
