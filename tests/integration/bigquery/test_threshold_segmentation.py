"""Tests for the ThresholdSegmentation class with BigQuery integration."""

import pytest

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()


@pytest.mark.parametrize(
    ("thresholds", "segments", "zero_value_handling"),
    [
        # Simple segmentation with separate zero segment
        # The original implementation expects the number of thresholds to equal the number of segments
        ([0.33, 0.66], ["Low", "High"], "separate_segment"),
        # Binary segmentation with including zeros in light category
        ([0.5], ["Below or Equal to Median"], "include_with_light"),
        # Test with excluding zero value customers
        ([0.33, 0.66], ["Low", "High"], "exclude"),
    ],
)
def test_threshold_segmentation_with_bigquery(
    transactions_table,
    thresholds,
    segments,
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
            thresholds=thresholds,
            segments=segments,
            value_col=cols.unit_spend,
            agg_func="sum",
            zero_value_customers=zero_value_handling,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"ThresholdSegmentation failed with thresholds={thresholds}, segments={segments}, zero_value_handling={zero_value_handling}: {e}",
        )
