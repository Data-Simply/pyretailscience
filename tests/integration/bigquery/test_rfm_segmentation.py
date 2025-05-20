"""Integration tests for the RFMSegmentation class using BigQuery."""

import datetime

import pytest

from pyretailscience.segmentation.rfm import RFMSegmentation


@pytest.mark.parametrize(
    "current_date",
    [None, "2023-12-31", datetime.date(2023, 6, 30)],
)
def test_rfm_segmentation_with_bigquery(
    transactions_table,
    current_date,
):
    """Test RFMSegmentation with data fetched from BigQuery.

    This parameterized test verifies that the RFMSegmentation class can be initialized
    and process data from BigQuery using different current_date parameters without throwing exceptions.
    """
    limited_table = transactions_table.limit(5000)

    result = RFMSegmentation(df=limited_table, current_date=current_date)
    assert result is not None

    df = result.df
    assert df is not None
