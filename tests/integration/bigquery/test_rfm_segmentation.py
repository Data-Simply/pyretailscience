"""Integration tests for the RFMSegmentation class using BigQuery."""

import datetime

import pytest

from pyretailscience.segmentation.rfm import RFMSegmentation


@pytest.mark.parametrize(
    (
        "current_date",
        "r_segments",
        "f_segments",
        "m_segments",
        "min_monetary",
        "max_monetary",
        "min_frequency",
        "max_frequency",
    ),
    [
        (None, 10, 10, 10, None, None, None, None),
        ("2023-12-31", 5, 5, 5, 0.0, 1000.0, 1, 50),
        (datetime.date(2023, 6, 30), [0.2, 0.4, 0.6, 0.8], [0.3, 0.6], [0.1, 0.5, 0.9], 10.0, 5000.0, 2, 100),
    ],
)
def test_rfm_segmentation_bigquery_executes(
    transactions_table,
    current_date,
    r_segments,
    f_segments,
    m_segments,
    min_monetary,
    max_monetary,
    min_frequency,
    max_frequency,
):
    """Test RFMSegmentation with data fetched from BigQuery.

    This parameterized test verifies that the RFMSegmentation class can be initialized
    and process data from BigQuery using different current_date parameters without throwing exceptions.
    """
    limited_table = transactions_table.limit(1000)

    result = RFMSegmentation(
        df=limited_table,
        current_date=current_date,
        r_segments=r_segments,
        f_segments=f_segments,
        m_segments=m_segments,
        min_monetary=min_monetary,
        max_monetary=max_monetary,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )
    assert result is not None

    df = result.df
    assert df is not None
