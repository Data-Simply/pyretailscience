"""Tests for the filter_and_label_by_condition function with BigQuery integration."""

from pyretailscience.utils.filter_and_label import filter_and_label_by_condition

SPEND_VALUE = 100


def test_filter_and_label_by_condition_with_bigquery(transactions_table):
    """Test filter_and_label_by_condition with data fetched from BigQuery.

    This test verifies that filter_and_label_by_condition can process data directly from
    a BigQuery connection using Ibis without throwing exceptions.
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
