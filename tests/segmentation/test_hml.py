"""Tests for the HMLSegmentation class."""

import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper, get_option, option_context
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

    @pytest.mark.parametrize(
        ("zero_value_customers", "value_col", "expected_segments"),
        [
            (None, None, {1: "Heavy", 2: "Light", 3: "Zero", 4: "Medium", 5: "Light"}),
            ("exclude", "custom_spend_col", {1: "Heavy", 2: "Light", 4: "Medium", 5: "Light"}),
            ("include_with_light", "custom_spend_col", {1: "Heavy", 2: "Light", 3: "Light", 4: "Medium", 5: "Light"}),
        ],
    )
    def test_with_custom_column_names(self, base_df, zero_value_customers, value_col, expected_segments):
        """Test that HMLSegmentation works correctly with completely renamed columns."""
        custom_columns = {
            "column.customer_id": "custom_cust_id",
            "column.unit_spend": "custom_spend_col",
        }

        rename_mapping = {
            get_option("column.customer_id"): custom_columns["column.customer_id"],
            cols.unit_spend: custom_columns["column.unit_spend"],
        }

        custom_df = base_df.rename(columns=rename_mapping)

        with option_context(*[item for pair in custom_columns.items() for item in pair]):
            hml_segmentation = HMLSegmentation(
                custom_df,
                zero_value_customers=zero_value_customers or "separate_segment",
                value_col=value_col,
            )
            result_df = hml_segmentation.df
            zero_value_customer_exclude_id = 3
            for customer_id, expected_segment in expected_segments.items():
                if zero_value_customers == "exclude" and customer_id == zero_value_customer_exclude_id:
                    assert customer_id not in result_df.index
                else:
                    assert result_df.loc[customer_id, "segment_name"] == expected_segment
