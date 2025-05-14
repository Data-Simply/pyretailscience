"""Integration tests for Cross Shop Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.cross_shop import CrossShop
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


@pytest.mark.parametrize(
    ("group_1_col", "group_2_col", "group_3_col", "value_col", "agg_func"),
    [
        # Test with 2 groups, different columns and aggregations
        ("brand_name", "category_1_name", None, "unit_spend", "sum"),
        ("category_0_name", "brand_name", None, "unit_quantity", "max"),
        ("category_1_name", "category_0_name", None, "unit_cost", "mean"),
        # Test with 3 groups
        ("brand_name", "category_0_name", "category_1_name", "unit_spend", "count"),
        ("category_0_name", "category_1_name", "brand_name", "unit_quantity", "mean"),
    ],
)
def test_cross_shop_with_bigquery(
    transactions_table,
    group_1_col,
    group_2_col,
    group_3_col,
    value_col,
    agg_func,
):
    """Test CrossShop with data fetched from BigQuery.

    This parameterized test verifies that CrossShop can be initialized
    and run with data from BigQuery using different combinations of group columns,
    value columns, and aggregation functions without throwing exceptions.
    """
    transactions_df = transactions_table.limit(5000)

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

    try:
        CrossShop(
            df=transactions_table,
            group_1_col=group_1_col,
            group_1_val=group_1_val,
            group_2_col=group_2_col,
            group_2_val=group_2_val,
            group_3_col=group_3_col,
            group_3_val=group_3_val,
            labels=labels,
            value_col=value_col,
            agg_func=agg_func,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"CrossShop failed with parameters {group_1_col}, {group_2_col}, {group_3_col}: {e}")
