"""Tests for the NLRSegmentation class."""

import ibis
import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.nlr import NLRSegmentation

cols = ColumnHelper()


class TestNLRSegmentation:
    """Tests for the NLRSegmentation class."""

    @pytest.fixture
    def transaction_df(self):
        """Return a DataFrame with transactions across two periods."""
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1001, 1003, 1004],
                cols.unit_spend: [100.00, 200.00, 150.00, 120.00, 180.00, 90.00],
                "year": [2023, 2023, 2023, 2024, 2024, 2024],
            },
        )

    def test_basic_segmentation(self, transaction_df):
        """Test customers are classified as New, Repeating, or Lapsed."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        result = seg.df

        # 1001 in both periods -> Repeating
        assert result.loc[1001, "segment_name"] == "Repeating"
        # 1002 only in P1 -> Lapsed
        assert result.loc[1002, "segment_name"] == "Lapsed"
        # 1003 in both periods -> Repeating
        assert result.loc[1003, "segment_name"] == "Repeating"
        # 1004 only in P2 -> New
        assert result.loc[1004, "segment_name"] == "New"

    def test_zero_spend_not_counted_as_bought(self):
        """Test that a customer with zero aggregated spend in a period is not considered to have bought."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1002],
                cols.unit_spend: [0.00, 50.00, 100.00],
                "quarter": ["Q1", "Q2", "Q1"],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="quarter",
            p1_value="Q1",
            p2_value="Q2",
        )
        result = seg.df

        # 1001 has zero spend in Q1 but positive in Q2 -> New
        assert result.loc[1001, "segment_name"] == "New"
        # 1002 only in Q1 with positive spend -> Lapsed
        assert result.loc[1002, "segment_name"] == "Lapsed"

    def test_negative_spend_not_counted_as_bought(self):
        """Test that a customer with negative aggregated spend in a period is not considered to have bought."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1001, 1001],
                cols.unit_spend: [-50.00, 30.00, 80.00],
                "period": ["P1", "P1", "P2"],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )
        result = seg.df

        # 1001 has negative aggregated spend (-50+30=-20) in P1 but positive in P2 -> New
        assert result.loc[1001, "segment_name"] == "New"

    def test_raises_on_missing_customer_id_column(self):
        """Test that ValueError is raised when customer_id column is missing."""
        df = pd.DataFrame(
            {
                cols.unit_spend: [100.00],
                "year": [2023],
            },
        )
        with pytest.raises(ValueError, match="missing"):
            NLRSegmentation(df=df, period_col="year", p1_value=2023, p2_value=2024)

    def test_raises_on_missing_value_column(self):
        """Test that ValueError is raised when value column is missing."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001],
                "year": [2023],
            },
        )
        with pytest.raises(ValueError, match="missing"):
            NLRSegmentation(df=df, period_col="year", p1_value=2023, p2_value=2024)

    def test_raises_on_missing_period_column(self):
        """Test that ValueError is raised when period column is missing."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001],
                cols.unit_spend: [100.00],
            },
        )
        with pytest.raises(ValueError, match="missing"):
            NLRSegmentation(df=df, period_col="year", p1_value=2023, p2_value=2024)

    def test_raises_on_invalid_p1_value(self):
        """Test that ValueError is raised when p1_value is not found in period_col."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001],
                cols.unit_spend: [100.00],
                "year": [2023],
            },
        )
        with pytest.raises(ValueError, match="p1_value"):
            NLRSegmentation(df=df, period_col="year", p1_value=2020, p2_value=2023)

    def test_raises_on_invalid_p2_value(self):
        """Test that ValueError is raised when p2_value is not found in period_col."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001],
                cols.unit_spend: [100.00],
                "year": [2023],
            },
        )
        with pytest.raises(ValueError, match="p2_value"):
            NLRSegmentation(df=df, period_col="year", p1_value=2023, p2_value=2025)

    def test_alternate_value_col(self):
        """Test segmentation with a non-default value column."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1001],
                cols.unit_qty: [5, 10, 8],
                "year": [2023, 2023, 2024],
            },
        )
        seg = NLRSegmentation(
            df=df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
            value_col=cols.unit_qty,
        )
        result = seg.df

        assert result.loc[1001, "segment_name"] == "Repeating"
        assert result.loc[1002, "segment_name"] == "Lapsed"

    def test_input_dataframe_not_mutated(self, transaction_df):
        """Test that the original DataFrame is not modified."""
        original_df = transaction_df.copy()
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        _ = seg.df
        assert original_df.equals(transaction_df)

    def test_table_property_returns_ibis_table(self, transaction_df):
        """Test that the table property returns an ibis Table."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        assert isinstance(seg.table, ibis.expr.types.Table)

    def test_accepts_ibis_table_input(self, transaction_df):
        """Test that an ibis Table can be passed directly as input."""
        table = ibis.memtable(transaction_df)
        seg = NLRSegmentation(
            df=table,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        result = seg.df

        assert result.loc[1001, "segment_name"] == "Repeating"
        assert result.loc[1002, "segment_name"] == "Lapsed"
        assert result.loc[1004, "segment_name"] == "New"

    def test_df_property_caches_result(self, transaction_df):
        """Test that the df property returns the same cached DataFrame on subsequent calls."""
        seg = NLRSegmentation(
            df=transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
        )
        first_call = seg.df
        second_call = seg.df
        assert first_call is second_call


class TestNLRSegmentationGroupCol:
    """Tests for NLRSegmentation group_col functionality."""

    @pytest.fixture
    def store_transaction_df(self):
        """Return a DataFrame with transactions across stores and periods."""
        return pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1001, 1003, 1001, 1002, 1003, 1004],
                cols.unit_spend: [100.00, 200.00, 150.00, 80.00, 120.00, 90.00, 70.00, 60.00],
                cols.store_id: [2001, 2001, 2001, 2001, 2002, 2002, 2002, 2002],
                "year": [2023, 2023, 2024, 2024, 2023, 2023, 2024, 2024],
            },
        )

    def test_segments_calculated_within_each_group(self, store_transaction_df):
        """Test that lapse segments are calculated independently within each group."""
        seg = NLRSegmentation(
            df=store_transaction_df,
            period_col="year",
            p1_value=2023,
            p2_value=2024,
            group_col=cols.store_id,
        )
        result = seg.df.sort_index()

        # Store 2001: 1001 in both->Repeating, 1002 P1 only->Lapsed, 1003 P2 only->New
        assert result.loc[(1001, 2001), "segment_name"] == "Repeating"
        assert result.loc[(1002, 2001), "segment_name"] == "Lapsed"
        assert result.loc[(1003, 2001), "segment_name"] == "New"

        # Store 2002: 1001 P1 only->Lapsed, 1002 P1 only->Lapsed, 1003 P2 only->New, 1004 P2 only->New
        assert result.loc[(1001, 2002), "segment_name"] == "Lapsed"
        assert result.loc[(1002, 2002), "segment_name"] == "Lapsed"
        assert result.loc[(1003, 2002), "segment_name"] == "New"
        assert result.loc[(1004, 2002), "segment_name"] == "New"

    def test_raises_on_missing_group_column(self):
        """Test that ValueError is raised when group_col column is missing from the DataFrame."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1001],
                cols.unit_spend: [100.00],
                "year": [2023],
            },
        )
        with pytest.raises(ValueError, match="missing"):
            NLRSegmentation(
                df=df,
                period_col="year",
                p1_value=2023,
                p2_value=2024,
                group_col=cols.store_id,
            )
