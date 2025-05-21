"""Tests for the date utility functions with BigQuery integration."""

from datetime import UTC, datetime

from pyretailscience.utils.date import filter_and_label_by_periods


def test_filter_and_label_by_periods_with_bigquery(transactions_table):
    """Test filter_and_label_by_periods with data using Ibis.

    This test verifies that filter_and_label_by_periods can process data
    through an Ibis without throwing exceptions.
    """
    limited_table = transactions_table.limit(1000)
    period_ranges = {
        "Q1": (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 3, 31, tzinfo=UTC)),
        "Q2": (datetime(2023, 4, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    }
    result = filter_and_label_by_periods(limited_table, period_ranges)

    assert result is not None

    df = result.execute()
    assert df is not None
