"""Integration tests for the HMLSegmentation class using BigQuery."""

import pytest

from pyretailscience.segmentation.hml import HMLSegmentation


@pytest.mark.parametrize(
    ("value_col", "agg_func", "zero_value_customers"),
    [
        ("unit_spend", "sum", "separate_segment"),
        ("unit_quantity", "sum", "include_with_light"),
        ("unit_cost", "mean", "exclude"),
        ("unit_spend", "mean", "separate_segment"),
    ],
)
def test_hml_segmentation_with_bigquery(
    transactions_table,
    value_col,
    agg_func,
    zero_value_customers,
):
    """Test HMLSegmentation with data fetched from BigQuery.

    This parameterized test verifies that HMLSegmentation can be initialized
    and process data from BigQuery using different combinations of value columns,
    aggregation functions, and zero-value handling without throwing exceptions.
    """
    limited_transactions = transactions_table.limit(5000)

    try:
        HMLSegmentation(
            df=limited_transactions,
            value_col=value_col,
            agg_func=agg_func,
            zero_value_customers=zero_value_customers,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"HMLSegmentation failed with value_col={value_col}, agg_func={agg_func}: {e}")
