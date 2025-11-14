"""Unified integration tests for filter_and_label_by_condition function with multiple database backends."""

from pyretailscience.utils.filter_and_label import filter_and_label_by_condition

SPEND_VALUE = 100


def test_filter_and_label_by_condition_integration(transactions_table):
    """Integration test for filter_and_label_by_condition using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
    """
    limited_table = transactions_table.limit(1000)
    conditions = {
        "high_spend": limited_table["unit_spend"] > SPEND_VALUE,
        "low_spend": limited_table["unit_spend"] <= SPEND_VALUE,
    }

    result = filter_and_label_by_condition(limited_table, conditions)
    assert result is not None

    df = result.execute()
    assert df is not None
