"""Integration tests for the HMLSegmentation class using BigQuery."""

import pytest

from pyretailscience.segmentation.hml import HMLSegmentation


@pytest.mark.parametrize(
    "zero_value_customers",
    ["separate_segment", "include_with_light", "exclude"],
)
def test_hml_segmentation_with_bigquery(
    transactions_table,
    zero_value_customers,
):
    """Test HMLSegmentation with data fetched from BigQuery.

    This parameterized test verifies that HMLSegmentation can be initialized
    and process data from BigQuery using zero-value handling without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(5000)

    result = HMLSegmentation(
        df=limited_transactions,
        value_col="unit_cost",
        agg_func="mean",
        zero_value_customers=zero_value_customers,
    )
    assert result is not None

    df = result.df
    assert df is not None
