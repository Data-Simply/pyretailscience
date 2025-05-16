"""Tests for the ThresholdSegmentation class with BigQuery integration."""

import pytest

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()


@pytest.mark.parametrize(
    "zero_value_handling",
    ["separate_segment", "exclude", "include_with_light"],
)
def test_threshold_segmentation_with_bigquery(
    transactions_table,
    zero_value_handling,
):
    """Test ThresholdSegmentation with data fetched from BigQuery.

    This test verifies that ThresholdSegmentation can process data directly from
    a BigQuery connection using Ibis without throwing exceptions.
    """
    try:
        limited_table = transactions_table.limit(1000)

        ThresholdSegmentation(
            df=limited_table,
            thresholds=[0.33, 0.66],
            segments=["Low", "High"],
            value_col=cols.unit_spend,
            agg_func="mean",
            zero_value_customers=zero_value_handling,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"ThresholdSegmentation failed with zero_value_handling={zero_value_handling}: {e}",
        )
