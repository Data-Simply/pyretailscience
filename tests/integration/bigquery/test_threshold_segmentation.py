"""Tests for the ThresholdSegmentation class with BigQuery integration."""

import pytest

from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()

MIN_BIGQUERY_ROWS = 4
MIN_TEST_CUSTOMERS = 5


class TestThresholdSegmentation:
    """Tests for the ThresholdSegmentation class with BigQuery integration."""

    def test_correct_segmentation(self, transactions_table):
        """Test that the method correctly segments customers based on given thresholds and segments."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(1000)

        df = query.execute()

        if len(df) < MIN_BIGQUERY_ROWS:
            pytest.skip("Not enough data in BigQuery for this test")

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

        assert "Low" in result_df["segment_name"].values
        assert "High" in result_df["segment_name"].values

        customer_totals = df.groupby(get_option("column.customer_id"))[cols.unit_spend].sum()
        median_spend = customer_totals.median()

        low_spender = customer_totals[customer_totals < median_spend].index[0]
        low_spender_segment = result_df.loc[low_spender, "segment_name"]
        assert low_spender_segment == "Low"

        high_spender = customer_totals[customer_totals > median_spend].index[0]
        high_spender_segment = result_df.loc[high_spender, "segment_name"]
        assert high_spender_segment == "High"

    def test_single_customer(self, transactions_table):
        """Test that the method correctly handles a DataFrame with only one customer."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(1)

        df = query.execute()

        thresholds = [0.5, 1]
        segments = ["Low"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(
                df=df,
                thresholds=thresholds,
                segments=segments,
            )

    def test_correct_aggregation_function(self, transactions_table):
        """Test that the correct aggregation function is applied for product_id custom segmentation."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.product_id,
        ).limit(1000)

        df = query.execute()

        value_col = "product_id"
        agg_func = "nunique"

        top_customers = df["customer_id"].value_counts().head(5).index.tolist()
        test_df = df[df["customer_id"].isin(top_customers)]

        my_seg = ThresholdSegmentation(
            df=test_df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=[0.2, 0.8, 1],
            segments=["Low", "Medium", "High"],
            zero_value_customers="separate_segment",
        )

        assert len(my_seg.df) == len(top_customers)
        assert set(my_seg.df["segment_name"].unique()).issubset({"Low", "Medium", "High", "Zero"})

        customer_product_counts = test_df.groupby(cols.customer_id)["product_id"].nunique()

        for customer_id in top_customers:
            segmented_value = my_seg.df.loc[customer_id, "product_id"]
            actual_value = customer_product_counts[customer_id]
            assert segmented_value == actual_value

    def test_correctly_checks_segment_data(self, transactions_table):
        """Test that the method correctly merges segment data back into the original DataFrame."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(1000)

        df = query.execute()

        customers_with_spend = df.groupby(get_option("column.customer_id"))[cols.unit_spend].sum()
        customers_with_zero = customers_with_spend[customers_with_spend == 0].index.tolist()
        customers_with_nonzero = customers_with_spend[customers_with_spend > 0].index.tolist()

        test_customers = customers_with_zero[:2] + customers_with_nonzero[:3]
        if len(test_customers) < MIN_TEST_CUSTOMERS:
            test_customers = customers_with_spend.index[:5].tolist()

        test_df = df[df[get_option("column.customer_id")].isin(test_customers)]

        value_col = cols.unit_spend
        agg_func = "sum"
        thresholds = [0.33, 0.66, 1]
        segments = ["Low", "Medium", "High"]
        zero_value_customers = "separate_segment"

        threshold_seg = ThresholdSegmentation(
            df=test_df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
        )

        segmented_df = threshold_seg.add_segment(test_df)

        assert "segment_name" in segmented_df.columns
        assert segmented_df["segment_name"].isnull().sum() == 0

        zero_spend_customers = test_df.groupby(get_option("column.customer_id"))[cols.unit_spend].sum()
        zero_spend_customers = zero_spend_customers[zero_spend_customers == 0].index

        for customer in zero_spend_customers:
            customer_rows = segmented_df[segmented_df[get_option("column.customer_id")] == customer]
            if not customer_rows.empty:
                customer_segment = customer_rows["segment_name"].values[0]
                assert customer_segment == "Zero", (
                    f"Expected 'Zero' segment for customer {customer} with zero spend, got {customer_segment}"
                )

    def test_handles_dataframe_with_duplicate_customer_id_entries(self, transactions_table):
        """Test that the method correctly handles a DataFrame with duplicate customer_id entries."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(1000)

        df = query.execute()

        customer_counts = df[get_option("column.customer_id")].value_counts()
        customers_with_dupes = customer_counts[customer_counts > 1].index[:3]

        if len(customers_with_dupes) == 0:
            pytest.skip("No customers with multiple transactions found in dataset")

        test_df = df[df[get_option("column.customer_id")].isin(customers_with_dupes)]

        my_seg = ThresholdSegmentation(
            df=test_df,
            value_col=cols.unit_spend,
            agg_func="sum",
            thresholds=[0.5, 0.8, 1],
            segments=["Light", "Medium", "Heavy"],
            zero_value_customers="include_with_light",
        )

        result_df = my_seg.add_segment(test_df)
        assert len(result_df) == len(test_df)

        for customer in customers_with_dupes:
            customer_rows = result_df[result_df[get_option("column.customer_id")] == customer]
            segments = customer_rows["segment_name"].unique()
            assert len(segments) == 1, f"Customer {customer} has multiple segments: {segments}"

    def test_thresholds_not_unique(self, transactions_table):
        """Test that the method raises an error when the thresholds are not unique."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(100)

        df = query.execute()

        thresholds = [0.5, 0.5, 0.8, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_few_segments(self, transactions_table):
        """Test that the method raises an error when there are too few/many segments for the number of thresholds."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(100)

        df = query.execute()

        thresholds = [0.4, 0.6, 0.8, 1]
        segments = ["Low", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_few_thresholds(self, transactions_table):
        """Test that the method raises an error when there are too few/many thresholds for the number of segments."""
        query = transactions_table.select(
            transactions_table.customer_id,
            transactions_table.unit_spend,
        ).limit(100)

        df = query.execute()

        thresholds = [0.4, 1]
        segments = ["Low", "Medium", "High"]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        thresholds = [0.2, 0.5, 0.6, 0.8, 1]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)
