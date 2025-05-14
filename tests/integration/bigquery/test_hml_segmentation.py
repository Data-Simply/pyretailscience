"""Integration tests for the HMLSegmentation class using BigQuery."""

import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.segmentation.hml import HMLSegmentation

cols = ColumnHelper()


class TestHMLSegmentationBigQuery:
    """Integration tests for the HMLSegmentation class using BigQuery data."""

    @pytest.fixture
    def customers_spend_df(self, transactions_table):
        """Create a DataFrame with customer spend data from BigQuery.

        This aggregates transaction data to customer level for HML segmentation.
        """
        customer_spend_query = transactions_table.group_by(get_option("column.customer_id")).aggregate(
            total_spend=transactions_table[cols.unit_spend].sum(),
        )

        customer_spend_df = customer_spend_query.execute().reset_index()

        if "index" in customer_spend_df.columns:
            customer_spend_df = customer_spend_df.drop(columns=["index"])

        return customer_spend_df.rename(columns={"total_spend": cols.unit_spend})

    def test_handles_zero_spend_customers_are_excluded_in_result(self, customers_spend_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is 'exclude'."""
        if not (customers_spend_df[cols.unit_spend] == 0).any():
            zero_customer = pd.DataFrame(
                {
                    get_option("column.customer_id"): [999999],
                    cols.unit_spend: [0],
                },
            )
            customers_spend_df = pd.concat([customers_spend_df, zero_customer], ignore_index=True)

        zero_spend_customer_id = customers_spend_df[customers_spend_df[cols.unit_spend] == 0][
            get_option("column.customer_id")
        ].iloc[0]

        hml_segmentation = HMLSegmentation(customers_spend_df, zero_value_customers="exclude")
        result_df = hml_segmentation.df

        assert zero_spend_customer_id not in result_df.index.values

        assert "Heavy" in result_df["segment_name"].values
        assert "Medium" in result_df["segment_name"].values
        assert "Light" in result_df["segment_name"].values

    def test_handles_zero_spend_customers_include_with_light(self, customers_spend_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is 'include_with_light'."""
        if not (customers_spend_df[cols.unit_spend] == 0).any():
            zero_customer = pd.DataFrame(
                {
                    get_option("column.customer_id"): [999999],
                    cols.unit_spend: [0],
                },
            )
            customers_spend_df = pd.concat([customers_spend_df, zero_customer], ignore_index=True)

        zero_spend_customer_id = customers_spend_df[customers_spend_df[cols.unit_spend] == 0][
            get_option("column.customer_id")
        ].iloc[0]

        hml_segmentation = HMLSegmentation(customers_spend_df, zero_value_customers="include_with_light")
        result_df = hml_segmentation.df

        zero_customer_row = (
            result_df.loc[[zero_spend_customer_id]] if zero_spend_customer_id in result_df.index else pd.DataFrame()
        )

        assert not zero_customer_row.empty
        assert zero_customer_row["segment_name"].iloc[0] == "Light"

    def test_handles_zero_spend_customers_separate_segment(self, customers_spend_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is 'separate_segment'."""
        if not (customers_spend_df[cols.unit_spend] == 0).any():
            zero_customer = pd.DataFrame(
                {
                    get_option("column.customer_id"): [999999],
                    cols.unit_spend: [0],
                },
            )
            customers_spend_df = pd.concat([customers_spend_df, zero_customer], ignore_index=True)

        zero_spend_customer_id = customers_spend_df[customers_spend_df[cols.unit_spend] == 0][
            get_option("column.customer_id")
        ].iloc[0]

        hml_segmentation = HMLSegmentation(customers_spend_df, zero_value_customers="separate_segment")
        result_df = hml_segmentation.df

        zero_customer_row = (
            result_df.loc[[zero_spend_customer_id]] if zero_spend_customer_id in result_df.index else pd.DataFrame()
        )

        assert not zero_customer_row.empty
        assert zero_customer_row["segment_name"].iloc[0] == "Zero"

    def test_raises_value_error_if_required_columns_missing(self, customers_spend_df):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        with pytest.raises(ValueError):
            HMLSegmentation(customers_spend_df.drop(columns=[get_option("column.customer_id")]))

    def test_input_dataframe_not_changed(self, customers_spend_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = customers_spend_df.copy()

        hml_segmentation = HMLSegmentation(customers_spend_df)
        _ = hml_segmentation.df

        pd.testing.assert_frame_equal(original_df, customers_spend_df)

    def test_alternate_value_col(self, transactions_table):
        """Test that the method correctly segments a DataFrame with an alternate value column."""
        customer_qty_query = transactions_table.group_by(get_option("column.customer_id")).aggregate(
            total_qty=transactions_table[cols.unit_qty].sum(),
        )

        customer_qty_df = customer_qty_query.execute().reset_index()

        if "index" in customer_qty_df.columns:
            customer_qty_df = customer_qty_df.drop(columns=["index"])

        customer_qty_df = customer_qty_df.rename(columns={"total_qty": cols.unit_qty})

        hml_segmentation = HMLSegmentation(customer_qty_df, value_col=cols.unit_qty)
        result_df = hml_segmentation.df

        assert "Heavy" in result_df["segment_name"].values
        assert "Medium" in result_df["segment_name"].values
        assert "Light" in result_df["segment_name"].values
