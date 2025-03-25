"""Tests for the ThresholdSegmentation class."""

import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()


class TestThresholdSegmentation:
    """Tests for the ThresholdSegmentation class."""

    def test_correct_segmentation(self):
        """Test that the method correctly segments customers based on given thresholds and segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4],
                cols.unit_spend: [100, 200, 300, 400],
            },
        )
        thresholds = [0.5, 1]
        segments = ["Low", "High"]
        seg = ThresholdSegmentation(
            df=df,
            thresholds=thresholds,
            segments=segments,
            value_col=cols.unit_spend,
            zero_value_customers="exclude",
        )
        result_df = seg.df
        assert result_df.loc[1, "segment_name"] == "Low"
        assert result_df.loc[2, "segment_name"] == "Low"
        assert result_df.loc[3, "segment_name"] == "High"
        assert result_df.loc[4, "segment_name"] == "High"

    def test_single_customer(self):
        """Test that the method correctly segments a DataFrame with only one customer."""
        df = pd.DataFrame({get_option("column.customer_id"): [1], cols.unit_spend: [100]})
        thresholds = [0.5, 1]
        segments = ["Low"]
        with pytest.raises(ValueError):
            ThresholdSegmentation(
                df=df,
                thresholds=thresholds,
                segments=segments,
            )

    def test_correct_aggregation_function(self):
        """Test that the correct aggregation function is applied for product_id custom segmentation."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
                "product_id": [3, 4, 4, 6, 1, 5, 7, 2, 2, 3, 2, 3, 4, 1],
            },
        )
        value_col = "product_id"
        agg_func = "nunique"

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=[0.2, 0.8, 1],
            segments=["Low", "Medium", "High"],
            zero_value_customers="separate_segment",
        )

        expected_result = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                "product_id": [1, 4, 2, 2, 3],
                "segment_name": ["Low", "High", "Medium", "Medium", "Medium"],
            },
        )
        pd.testing.assert_frame_equal(my_seg.df.sort_values(cols.customer_id).reset_index(), expected_result)

    def test_correctly_checks_segment_data(self):
        """Test that the method correctly merges segment data back into the original DataFrame."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 0, 150, 0],
            },
        )
        value_col = cols.unit_spend
        agg_func = "sum"
        thresholds = [0.33, 0.66, 1]
        segments = ["Low", "Medium", "High"]
        zero_value_customers = "separate_segment"

        # Create ThresholdSegmentation instance
        threshold_seg = ThresholdSegmentation(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
        )

        # Call add_segment method
        segmented_df = threshold_seg.add_segment(df)

        # Assert the correct segment_name
        expected_df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 0, 150, 0],
                "segment_name": ["Low", "High", "Zero", "Medium", "Zero"],
            },
        )
        pd.testing.assert_frame_equal(segmented_df, expected_df)

    def test_handles_dataframe_with_duplicate_customer_id_entries(self):
        """Test that the method correctly handles a DataFrame with duplicate customer_id entries."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 1, 2, 3],
                cols.unit_spend: [100, 200, 300, 150, 250, 350],
            },
        )

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=cols.unit_spend,
            agg_func="sum",
            thresholds=[0.5, 0.8, 1],
            segments=["Light", "Medium", "Heavy"],
            zero_value_customers="include_with_light",
        )

        result_df = my_seg.add_segment(df)
        assert len(result_df) == len(df)

    def test_thresholds_not_unique(self):
        """Test that the method raises an error when the thresholds are not unique."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.5, 0.5, 0.8, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_few_segments(self):
        """Test that the method raises an error when there are too few/many segments for the number of thresholds."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 0.6, 0.8, 1]
        segments = ["Low", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_too_few_thresholds(self):
        """Test that the method raises an error when there are too few/many thresholds for the number of segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        thresholds = [0.2, 0.5, 0.6, 0.8, 1]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)
