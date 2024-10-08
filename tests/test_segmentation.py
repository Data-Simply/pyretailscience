"""Tests for the SegTransactionStats class."""

import pandas as pd
import pytest

from pyretailscience.options import get_option
from pyretailscience.segmentation import HMLSegmentation, SegTransactionStats, ThresholdSegmentation


class TestCalcSegStats:
    """Tests for the _calc_seg_stats method."""

    @pytest.fixture()
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 150, 300, 250],
                get_option("column.transaction_id"): [101, 102, 103, 104, 105],
                "segment_id": ["A", "B", "A", "B", "A"],
                get_option("column.unit_quantity"): [10, 20, 15, 30, 25],
            },
        )

    def test_correctly_calculates_revenue_transactions_customers_per_segment(self, base_df):
        """Test that the method correctly calculates at the transaction-item level."""
        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                get_option("column.agg.unit_spend"): [500.0, 500.0, 1000.0],
                get_option("column.agg.transaction_id"): [3, 2, 5],
                get_option("column.agg.customer_id"): [3, 2, 5],
                get_option("column.agg.unit_quantity"): [50, 50, 100],
                get_option("column.calc.spend_per_customer"): [166.666667, 250.0, 200.0],
                get_option("column.calc.spend_per_transaction"): [166.666667, 250.0, 200.0],
                get_option("column.calc.transactions_per_customer"): [1.0, 1.0, 1.0],
                f"customers_{get_option('column.suffix.percent')}": [0.6, 0.4, 1.0],
                get_option("column.calc.price_per_unit"): [10.0, 10.0, 10.0],
                get_option("column.calc.units_per_transaction"): [16.666667, 25.0, 20.0],
            },
        ).set_index("segment_name")

        segment_stats = SegTransactionStats._calc_seg_stats(base_df, "segment_id")
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_correctly_calculates_revenue_transactions_customers(self):
        """Test that the method correctly calculates at the transaction level."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 150, 300, 250],
                get_option("column.transaction_id"): [101, 102, 103, 104, 105],
                "segment_id": ["A", "B", "A", "B", "A"],
            },
        )

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "B", "Total"],
                get_option("column.agg.unit_spend"): [500.0, 500.0, 1000.0],
                get_option("column.agg.transaction_id"): [3, 2, 5],
                get_option("column.agg.customer_id"): [3, 2, 5],
                get_option("column.calc.spend_per_customer"): [166.666667, 250.0, 200.0],
                get_option("column.calc.spend_per_transaction"): [166.666667, 250.0, 200.0],
                get_option("column.calc.transactions_per_customer"): [1.0, 1.0, 1.0],
                f"customers_{get_option('column.suffix.percent')}": [0.6, 0.4, 1.0],
            },
        ).set_index("segment_name")

        segment_stats = SegTransactionStats._calc_seg_stats(df, "segment_id")
        pd.testing.assert_frame_equal(segment_stats, expected_output)

    def test_does_not_alter_original_dataframe(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()
        _ = SegTransactionStats._calc_seg_stats(base_df, "segment_id")

        pd.testing.assert_frame_equal(base_df, original_df)

    def test_handles_dataframe_with_one_segment(self, base_df):
        """Test that the method correctly handles a DataFrame with only one segment."""
        df = base_df.copy()
        df["segment_id"] = "A"

        expected_output = pd.DataFrame(
            {
                "segment_name": ["A", "Total"],
                get_option("column.agg.unit_spend"): [1000.0, 1000.0],
                get_option("column.agg.transaction_id"): [5, 5],
                get_option("column.agg.customer_id"): [5, 5],
                get_option("column.agg.unit_quantity"): [100, 100],
                get_option("column.calc.spend_per_customer"): [200.0, 200.0],
                get_option("column.calc.spend_per_transaction"): [200.0, 200.0],
                get_option("column.calc.transactions_per_customer"): [1.0, 1.0],
                f"customers_{get_option('column.suffix.percent')}": [1.0, 1.0],
                get_option("column.calc.price_per_unit"): [10.0, 10.0],
                get_option("column.calc.units_per_transaction"): [20.0, 20.0],
            },
        ).set_index("segment_name")

        segment_stats = SegTransactionStats._calc_seg_stats(df, "segment_id")
        pd.testing.assert_frame_equal(segment_stats, expected_output)


class TestThresholdSegmentation:
    """Tests for the ThresholdSegmentation class."""

    def test_correct_segmentation(self):
        """Test that the method correctly segments customers based on given thresholds and segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4],
                get_option("column.unit_spend"): [100, 200, 300, 400],
            },
        )
        thresholds = [0.5, 1]
        segments = {0: "Low", 1: "High"}
        seg = ThresholdSegmentation(
            df=df,
            thresholds=thresholds,
            segments=segments,
            value_col=get_option("column.unit_spend"),
            zero_value_customers="exclude",
        )
        result_df = seg.df
        assert result_df.loc[1, "segment_name"] == "Low"
        assert result_df.loc[2, "segment_name"] == "Low"
        assert result_df.loc[3, "segment_name"] == "High"
        assert result_df.loc[4, "segment_name"] == "High"

    def test_single_customer(self):
        """Test that the method correctly segments a DataFrame with only one customer."""
        df = pd.DataFrame({get_option("column.customer_id"): [1], get_option("column.unit_spend"): [100]})
        thresholds = [0.5, 1]
        segments = {0: "Low"}
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
                get_option("column.customer_id"): [1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
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
            segments={"A": "Low", "B": "Medium", "C": "High"},
            zero_value_customers="separate_segment",
        )

        expected_result = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                "product_id": [1, 4, 2, 2, 3],
                "segment_name": ["Low", "High", "Medium", "Medium", "Medium"],
                "segment_id": ["A", "C", "B", "B", "B"],
            },
        )
        expected_result["segment_id"] = pd.Categorical(
            expected_result["segment_id"],
            categories=["A", "B", "C"],
            ordered=True,
        )
        expected_result["segment_name"] = pd.Categorical(
            expected_result["segment_name"],
            categories=["Low", "Medium", "High"],
            ordered=True,
        )
        pd.testing.assert_frame_equal(my_seg.df.reset_index(), expected_result)

    def test_correctly_checks_segment_data(self):
        """Test that the method correctly merges segment data back into the original DataFrame."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 0, 150, 0],
            },
        )
        value_col = get_option("column.unit_spend")
        agg_func = "sum"
        thresholds = [0.33, 0.66, 1]
        segments = {"A": "Low", "B": "Medium", "C": "High"}
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

        # Assert the correct segment_name and segment_id
        expected_df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 0, 150, 0],
                "segment_name": ["Low", "High", "Zero", "Medium", "Zero"],
                "segment_id": ["A", "C", "Z", "B", "Z"],
            },
        )
        pd.testing.assert_frame_equal(segmented_df, expected_df)

    def test_handles_dataframe_with_duplicate_customer_id_entries(self):
        """Test that the method correctly handles a DataFrame with duplicate customer_id entries."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 1, 2, 3],
                get_option("column.unit_spend"): [100, 200, 300, 150, 250, 350],
            },
        )

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=get_option("column.unit_spend"),
            agg_func="sum",
            thresholds=[0.5, 0.8, 1],
            segments={"L": "Light", "M": "Medium", "H": "Heavy"},
            zero_value_customers="include_with_light",
        )

        result_df = my_seg.add_segment(df)
        assert len(result_df) == len(df)

    def test_correctly_maps_segment_names_to_segment_ids_with_fixed_thresholds(self):
        """Test that the method correctly maps segment names to segment IDs with fixed thresholds."""
        # Setup
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 300, 400, 500],
            },
        )
        value_col = get_option("column.unit_spend")
        agg_func = "sum"
        thresholds = [0.33, 0.66, 1]
        segments = {1: "Low", 2: "Medium", 3: "High"}
        zero_value_customers = "separate_segment"

        my_seg = ThresholdSegmentation(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
        )

        assert len(my_seg.df[["segment_id", "segment_name"]].drop_duplicates()) == len(segments)
        assert my_seg.df.set_index("segment_id")["segment_name"].to_dict() == segments

    def test_thresholds_not_unique(self):
        """Test that the method raises an error when the thresholds are not unique."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.5, 0.5, 0.8, 1]
        segments = {1: "Low", 2: "Medium", 3: "High"}

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_few_segments(self):
        """Test that the method raises an error when there are too few/many segments for the number of thresholds."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 0.6, 0.8, 1]
        segments = {1: "Low", 3: "High"}

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        segments = {1: "Low", 2: "Medium", 3: "High"}

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

    def test_thresholds_too_too_few_thresholds(self):
        """Test that the method raises an error when there are too few/many thresholds for the number of segments."""
        df = pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [100, 200, 300, 400, 500],
            },
        )
        thresholds = [0.4, 1]
        segments = {1: "Low", 2: "Medium", 3: "High"}

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)

        thresholds = [0.2, 0.5, 0.6, 0.8, 1]

        with pytest.raises(ValueError):
            ThresholdSegmentation(df, thresholds, segments)


class TestSegTransactionStats:
    """Tests for the SegTransactionStats class."""

    def test_handles_empty_dataframe_with_errors(self):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        df = pd.DataFrame(
            columns=[get_option("column.unit_spend"), get_option("column.transaction_id"), "segment_id", "quantity"],
        )

        with pytest.raises(ValueError):
            SegTransactionStats(df, "segment_id")


class TestHMLSegmentation:
    """Tests for the HMLSegmentation class."""

    @pytest.fixture()
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                get_option("column.customer_id"): [1, 2, 3, 4, 5],
                get_option("column.unit_spend"): [1000, 200, 0, 500, 300],
            },
        )

    def test_no_transactions(self):
        """Test that the method raises an error when there are no transactions."""
        data = {get_option("column.customer_id"): [], get_option("column.unit_spend"): []}
        df = pd.DataFrame(data)
        with pytest.raises(ValueError):
            HMLSegmentation(df)

    # Correctly handles zero spend customers when zero_value_customers is "exclude"
    def test_handles_zero_spend_customers_are_excluded_in_result(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "exclude"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="exclude")
        result_df = hml_segmentation.df

        zero_spend_customer_id = 3

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert zero_spend_customer_id not in result_df.index
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"

    # Correctly handles zero spend customers when zero_value_customers is "include_with_light"
    def test_handles_zero_spend_customers_include_with_light(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "include_with_light"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="include_with_light")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert result_df.loc[3, "segment_name"] == "Light"
        assert result_df.loc[3, "segment_id"] == "L"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"

    # Correctly handles zero spend customers when zero_value_customers is "separate_segment"
    def test_handles_zero_spend_customers_separate_segment(self, base_df):
        """Test that the method correctly handles zero spend customers when zero_value_customers is "separate_segment"."""
        hml_segmentation = HMLSegmentation(base_df, zero_value_customers="separate_segment")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert result_df.loc[3, "segment_name"] == "Zero"
        assert result_df.loc[3, "segment_id"] == "Z"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"

    # Raises ValueError if required columns are missing
    def test_raises_value_error_if_required_columns_missing(self, base_df):
        """Test that the method raises an error when the DataFrame is missing a required column."""
        with pytest.raises(ValueError):
            HMLSegmentation(base_df.drop(columns=[get_option("column.customer_id")]))

    # DataFrame with only one customer
    def test_segments_customer_single(self):
        """Test that the method correctly segments a DataFrame with only one customer."""
        data = {get_option("column.customer_id"): [1], get_option("column.unit_spend"): [0]}
        df = pd.DataFrame(data)
        with pytest.raises(ValueError):
            HMLSegmentation(df)

    # Validate that the input dataframe is not changed
    def test_input_dataframe_not_changed(self, base_df):
        """Test that the method does not alter the original DataFrame."""
        original_df = base_df.copy()

        hml_segmentation = HMLSegmentation(base_df)
        _ = hml_segmentation.df

        assert original_df.equals(base_df)  # Check if the original dataframe is not changed

    def test_alternate_value_col(self, base_df):
        """Test that the method correctly segments a DataFrame with an alternate value column."""
        base_df = base_df.rename(columns={get_option("column.unit_spend"): "quantity"})
        hml_segmentation = HMLSegmentation(base_df, value_col="quantity")
        result_df = hml_segmentation.df

        assert result_df.loc[1, "segment_name"] == "Heavy"
        assert result_df.loc[1, "segment_id"] == "H"
        assert result_df.loc[2, "segment_name"] == "Light"
        assert result_df.loc[2, "segment_id"] == "L"
        assert result_df.loc[4, "segment_name"] == "Medium"
        assert result_df.loc[4, "segment_id"] == "M"
        assert result_df.loc[5, "segment_name"] == "Light"
        assert result_df.loc[5, "segment_id"] == "L"
