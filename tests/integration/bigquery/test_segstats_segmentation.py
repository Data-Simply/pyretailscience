"""Integration tests for segmentation statistics using BigQuery data."""

import numpy as np
import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.segstats import SegTransactionStats

cols = ColumnHelper()

MIN_SEGMENT_COUNT = 3
MIN_DATE_SEGMENTS = 2


class TestBigQuerySegmentationStats:
    """Integration tests for segmentation statistics using BigQuery."""

    @pytest.fixture
    def transaction_df(self, transactions_table):
        """Convert BigQuery transactions table to DataFrame for testing."""
        df = transactions_table.execute()

        df["segment_name"] = np.where(df[cols.unit_spend] > df[cols.unit_spend].median(), "High", "Low")

        df["region"] = np.where(df["store_id"] % 3 == 0, "North", np.where(df["store_id"] % 3 == 1, "South", "East"))

        return df

    def test_bigquery_segment_transaction_stats(self, transaction_df):
        """Test segment statistics calculation with BigQuery data."""
        segment_stats = SegTransactionStats(transaction_df, "segment_name").df

        assert "segment_name" in segment_stats.columns
        assert "Total" in segment_stats["segment_name"].values
        assert "High" in segment_stats["segment_name"].values
        assert "Low" in segment_stats["segment_name"].values

        assert cols.agg_unit_spend in segment_stats.columns
        assert cols.agg_transaction_id in segment_stats.columns
        assert cols.agg_customer_id in segment_stats.columns
        assert cols.calc_spend_per_cust in segment_stats.columns
        assert cols.calc_spend_per_trans in segment_stats.columns
        assert cols.calc_trans_per_cust in segment_stats.columns

        total_row = segment_stats[segment_stats["segment_name"] == "Total"]
        segment_rows = segment_stats[segment_stats["segment_name"] != "Total"]

        assert total_row[cols.agg_unit_spend].values[0] == pytest.approx(segment_rows[cols.agg_unit_spend].sum())

        assert total_row[cols.agg_transaction_id].values[0] > 0
        assert total_row[cols.agg_customer_id].values[0] > 0

        total_transactions = len(transaction_df[cols.transaction_id].unique())
        assert total_row[cols.agg_transaction_id].values[0] == total_transactions

    def test_bigquery_segment_stats_without_total(self, transaction_df):
        """Test segment statistics without total row calculation."""
        segment_stats = SegTransactionStats(transaction_df, "segment_name", calc_total=False).df

        assert "Total" not in segment_stats["segment_name"].values

        assert set(segment_stats["segment_name"].unique()) == {"High", "Low"}

        for segment in ["High", "Low"]:
            segment_data = transaction_df[transaction_df["segment_name"] == segment]
            segment_row = segment_stats[segment_stats["segment_name"] == segment]

            expected_spend = segment_data[cols.unit_spend].sum()
            expected_trans_count = len(segment_data[cols.transaction_id].unique())
            expected_cust_count = len(segment_data[cols.customer_id].unique())

            assert segment_row[cols.agg_unit_spend].values[0] == pytest.approx(expected_spend, rel=1e-10)
            assert segment_row[cols.agg_transaction_id].values[0] == expected_trans_count
            assert segment_row[cols.agg_customer_id].values[0] == expected_cust_count

    def test_bigquery_multiple_segment_columns(self, transaction_df):
        """Test segment statistics with multiple segment columns."""
        seg_stats = SegTransactionStats(transaction_df, ["segment_name", "region"]).df

        assert "segment_name" in seg_stats.columns
        assert "region" in seg_stats.columns

        total_row = seg_stats[(seg_stats["segment_name"] == "Total") & (seg_stats["region"] == "Total")]
        assert len(total_row) == 1

        segment_region_rows = seg_stats[(seg_stats["segment_name"] != "Total") & (seg_stats["region"] != "Total")]
        assert len(segment_region_rows) > 0

        data_combinations = (
            transaction_df.groupby(["segment_name", "region"]).size().reset_index()[["segment_name", "region"]]
        )
        for _, row in data_combinations.iterrows():
            matching_rows = segment_region_rows[
                (segment_region_rows["segment_name"] == row["segment_name"])
                & (segment_region_rows["region"] == row["region"])
            ]
            assert len(matching_rows) == 1, f"Missing combination: {row['segment_name']}, {row['region']}"

    def test_bigquery_extra_aggregations(self, transaction_df):
        """Test segment statistics with additional custom aggregations."""
        extra_aggs = {
            "distinct_stores": ("store_id", "nunique"),
            "distinct_products": ("product_id", "nunique"),
            "avg_unit_cost": ("unit_cost", "mean"),
        }

        seg_stats = SegTransactionStats(
            transaction_df,
            "segment_name",
            extra_aggs=extra_aggs,
        ).df

        assert "distinct_stores" in seg_stats.columns
        assert "distinct_products" in seg_stats.columns
        assert "avg_unit_cost" in seg_stats.columns

        assert (seg_stats["distinct_stores"] > 0).all()
        assert (seg_stats["distinct_products"] > 0).all()

        for _, row in seg_stats[seg_stats["segment_name"] != "Total"].iterrows():
            segment = row["segment_name"]
            segment_data = transaction_df[transaction_df["segment_name"] == segment]

            expected_stores = segment_data["store_id"].nunique()
            expected_products = segment_data["product_id"].nunique()
            expected_avg_cost = segment_data["unit_cost"].mean()

            assert row["distinct_stores"] == expected_stores
            assert row["distinct_products"] == expected_products
            assert row["avg_unit_cost"] == pytest.approx(expected_avg_cost, rel=1e-10)

        total_row = seg_stats[seg_stats["segment_name"] == "Total"]
        assert total_row["distinct_stores"].values[0] == transaction_df["store_id"].nunique()
        assert total_row["distinct_products"].values[0] == transaction_df["product_id"].nunique()

    def test_plot_functionality(self, transaction_df):
        """Test that the plotting functionality works with BigQuery data."""
        seg_stats = SegTransactionStats(transaction_df, "segment_name")

        assert isinstance(seg_stats.df, pd.DataFrame)
        assert "segment_name" in seg_stats.df.columns
        assert len(seg_stats.df) >= MIN_SEGMENT_COUNT

        try:
            seg_stats.plot("spend")
            seg_stats.plot("transactions")
            seg_stats.plot("customers")
            plotting_worked = True
        except (ValueError, TypeError, RuntimeError) as e:
            plotting_worked = False
            pytest.fail(f"Plotting failed with error: {e!s}")

        assert plotting_worked, "Plotting functionality should not raise an error"

        seg_stats_no_total = SegTransactionStats(transaction_df, "segment_name", calc_total=False)
        try:
            seg_stats_no_total.plot("spend")
            plotting_no_total_worked = True
        except Exception as e:  # noqa: BLE001
            plotting_no_total_worked = False
            pytest.fail(f"Plotting with calc_total=False failed: {e!s}")

        assert plotting_no_total_worked, "Plotting with calc_total=False should work"

    def test_category_level_segmentation(self, transaction_df):
        """Test segment statistics with category-level segmentation."""
        transaction_df["category_segment"] = transaction_df["category_0_name"]
        seg_stats = SegTransactionStats(transaction_df, "category_segment").df

        assert len(seg_stats["category_segment"].unique()) > 1
        assert "Total" in seg_stats["category_segment"].values

        single_category = seg_stats[seg_stats["category_segment"] != "Total"].iloc[0]
        category_name = single_category["category_segment"]
        category_data = transaction_df[transaction_df["category_segment"] == category_name]
        expected_spend = category_data[cols.unit_spend].sum()
        expected_trans_count = len(category_data[cols.transaction_id].unique())
        expected_cust_count = len(category_data[cols.customer_id].unique())

        assert single_category[cols.agg_unit_spend] == pytest.approx(expected_spend, rel=1e-10)
        assert single_category[cols.agg_transaction_id] == expected_trans_count
        assert single_category[cols.agg_customer_id] == expected_cust_count

    def test_brand_level_segmentation(self, transaction_df):
        """Test segment statistics with brand-level segmentation."""
        if "brand_name" in transaction_df.columns:
            transaction_df["brand_segment"] = transaction_df["brand_name"].astype(str).str[:2]
        else:
            transaction_df["brand_segment"] = transaction_df["brand_id"].astype(str).str[:2]

        seg_stats = SegTransactionStats(transaction_df, "brand_segment").df
        assert len(seg_stats["brand_segment"].unique()) > 1
        brand_segments = seg_stats[seg_stats["brand_segment"] != "Total"]

        assert (brand_segments[cols.agg_unit_spend] > 0).all()
        assert (brand_segments[cols.customers_pct] > 0).all()
        assert (brand_segments[cols.customers_pct] <= 1).all()

        total_row = seg_stats[seg_stats["brand_segment"] == "Total"]
        assert total_row[cols.customers_pct].values[0] == pytest.approx(1.0)

    def test_date_based_segmentation(self, transaction_df):
        """Test segment statistics with date-based segmentation."""
        transaction_df["transaction_date"] = pd.to_datetime(transaction_df["transaction_date"])

        transaction_df["date_segment"] = transaction_df["transaction_date"].dt.strftime("%Y-%m")

        seg_stats = SegTransactionStats(transaction_df, "date_segment").df

        assert len(seg_stats["date_segment"].unique()) >= MIN_DATE_SEGMENTS
        assert "Total" in seg_stats["date_segment"].values

        month_segments = seg_stats[seg_stats["date_segment"] != "Total"]

        assert (month_segments[cols.agg_transaction_id] > 0).all()
        assert (month_segments[cols.calc_spend_per_cust] > 0).all()

        if len(month_segments) > 0:
            test_month = month_segments.iloc[0]["date_segment"]
            month_data = transaction_df[transaction_df["date_segment"] == test_month]

            expected_spend = month_data[cols.unit_spend].sum()
            expected_trans_count = len(month_data[cols.transaction_id].unique())

            month_row = month_segments[month_segments["date_segment"] == test_month]

            assert month_row[cols.agg_unit_spend].values[0] == pytest.approx(expected_spend, rel=1e-10)
            assert month_row[cols.agg_transaction_id].values[0] == expected_trans_count
