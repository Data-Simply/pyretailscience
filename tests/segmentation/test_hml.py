"""Tests for the HMLSegmentation class."""

import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.segmentation.hml import HMLSegmentation

cols = ColumnHelper()


class TestHMLSegmentation:
    """Tests for the HMLSegmentation class."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                cols.unit_spend: [1000, 200, 0, 500, 300],
            },
        )

    # Correctly handles zero spend customers when zero_value_customers is "exclude"
    def test_handles_zero_spend_customers_are_excluded_in_result(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "exclude"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="exclude")
        result_df = hml_segmentation.df

        zero_spend_customer_id = 3

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert zero_spend_customer_id not in result_df.index
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"

    # Correctly handles zero spend customers when zero_value_customers is "include_with_light"
    def test_handles_zero_spend_customers_include_with_light(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "include_with_light"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="include_with_light")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[3, "segment_name"] == "Light"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"

    # Correctly handles zero spend customers when zero_value_customers is "separate_segment"
    def test_handles_zero_spend_customers_separate_segment(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "separate_segment"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="separate_segment")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[3, "segment_name"] == "Zero"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"

    # Raises ValueError if required columns are missing
    def test_raises_value_error_if_required_columns_missing(self, base_df):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        with pytest.raises(ValueError):
            HMLSegmentation(base_df.drop(columns=[get_option("column.customer_id")]))

    # Validate that the input dataframe is not changed
    def test_input_dataframe_not_changed(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()

        hml_segmentation = HMLSegmentation(base_df)
        _ = hml_segmentation.df

        assert original_df.equals(base_df)  # Check if the original dataframe is not changed

    def test_alternate_value_col(self, base_df):
        """Test that the method correctly segments a DataFrame with an alternate value column."""
        base_df = base_df.rename(columns={cols.unit_spend: cols.unit_qty})
        hml_segmentation = HMLSegmentation(base_df, value_col=cols.unit_qty)
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[5, "segment_name"] == "Light"


class TestHMLSegmentationGroupCol:
    """Tests for HMLSegmentation group_col functionality."""

    @pytest.fixture
    def store_transaction_df(self):
        """Return a DataFrame with customer transactions across multiple stores for HML testing."""
        # 5 customers per store to test HML distribution
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.unit_spend: [
                    # Store 2001: 1001 highest, 1005 lowest
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    100.00,
                    # Store 2002: 1005 highest, 1001 lowest (reversed)
                    100.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                cols.store_id: [2001] * 5 + [2002] * 5,
            },
        )

    def test_hml_segments_calculated_within_each_store(self, store_transaction_df):
        """Test that HML segments are calculated independently within each store."""
        hml = HMLSegmentation(
            df=store_transaction_df,
            value_col=cols.unit_spend,
            group_col=cols.store_id,
            zero_value_customers="exclude",
        )

        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.store_id: [2001] * 5 + [2002] * 5,
                cols.unit_spend: [
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    100.00,
                    100.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                # Store 2001: 1001=Heavy, 1002=Medium, 1003/1004/1005=Light
                # Store 2002: 1005=Heavy, 1004=Medium, 1001/1002/1003=Light
                "segment_name": [
                    "Heavy",
                    "Medium",
                    "Light",
                    "Light",
                    "Light",
                    "Light",
                    "Light",
                    "Light",
                    "Medium",
                    "Heavy",
                ],
            },
        ).set_index([cols.customer_id, cols.store_id])

        pd.testing.assert_frame_equal(hml.df.sort_index(), expected_df.sort_index())

    def test_hml_with_zero_value_and_group_col(self):
        """Test HML segmentation handles zero-value customers correctly within groups."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.unit_spend: [
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    0.00,
                    0.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                cols.store_id: [2001] * 5 + [2002] * 5,
            },
        )

        hml = HMLSegmentation(
            df=df,
            value_col=cols.unit_spend,
            group_col=cols.store_id,
            zero_value_customers="separate_segment",
        )

        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1001, 1002, 1003, 1004, 1005],
                cols.store_id: [2001] * 5 + [2002] * 5,
                cols.unit_spend: [
                    1000.00,
                    500.00,
                    300.00,
                    200.00,
                    0.00,
                    0.00,
                    200.00,
                    300.00,
                    500.00,
                    1000.00,
                ],
                "segment_name": [
                    "Heavy",
                    "Medium",
                    "Light",
                    "Light",
                    "Zero",
                    "Zero",
                    "Light",
                    "Light",
                    "Medium",
                    "Heavy",
                ],
            },
        ).set_index([cols.customer_id, cols.store_id])

        pd.testing.assert_frame_equal(hml.df.sort_index(), expected_df.sort_index())
