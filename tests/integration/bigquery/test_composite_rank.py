"""Integration tests for Composite Rank Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.composite_rank import CompositeRank


class TestCompositeRank:
    """Tests for the CompositeRank class."""

    @pytest.fixture(scope="class")
    def test_transactions_df(self, transactions_table):
        """Fetch test transactions data from BigQuery and convert to DataFrame.

        This fixture assumes a table with columns like product_id, spend, customers, etc.
        Modify the query and column names as per your actual BigQuery table structure.
        """
        df = transactions_table.to_pandas()

        if "spend_per_customer" not in df.columns:
            df["spend_per_customer"] = df["unit_spend"] / df["customer_id"]

        return df

    def test_composite_rank_with_bigquery_data(self, test_transactions_df):
        """Test CompositeRank functionality with real BigQuery data.

        This test demonstrates using CompositeRank with BigQuery-sourced data.
        """
        rank_cols = [
            ("unit_spend", "desc"),
            ("customer_id", "desc"),
            ("spend_per_customer", "desc"),
        ]

        cr = CompositeRank(
            df=test_transactions_df,
            rank_cols=rank_cols,
            agg_func="mean",
            ignore_ties=False,
        )

        assert "composite_rank" in cr.df.columns
        assert len(cr.df) > 0

        expected_rank_columns = [
            "unit_spend_rank",
            "customer_id_rank",
            "spend_per_customer_rank",
            "composite_rank",
        ]
        for col in expected_rank_columns:
            assert col in cr.df.columns

    def test_different_agg_functions_with_bigquery(self, test_transactions_df):
        """Test different aggregation functions with BigQuery data."""
        agg_functions = ["mean", "sum", "min", "max"]

        rank_cols = [
            ("unit_spend", "desc"),
            ("customer_id", "desc"),
            ("spend_per_customer", "desc"),
        ]

        for agg_func in agg_functions:
            cr = CompositeRank(
                df=test_transactions_df,
                rank_cols=rank_cols,
                agg_func=agg_func,
                ignore_ties=False,
            )

            assert "composite_rank" in cr.df.columns
            assert len(cr.df) > 0

    def test_ignore_ties_with_bigquery(self, test_transactions_df):
        """Test tie-breaking behavior with BigQuery data."""
        rank_cols = [("unit_spend", "desc")]

        cr_with_ties = CompositeRank(
            df=test_transactions_df,
            rank_cols=rank_cols,
            agg_func="mean",
            ignore_ties=False,
        )

        cr_no_ties = CompositeRank(
            df=test_transactions_df,
            rank_cols=rank_cols,
            agg_func="mean",
            ignore_ties=True,
        )

        assert "unit_spend_rank" in cr_with_ties.df.columns
        assert "unit_spend_rank" in cr_no_ties.df.columns

    def test_ibis_table_input(self, transactions_table):
        """Explicitly test Ibis table input for CompositeRank."""
        cr = CompositeRank(
            df=transactions_table,
            rank_cols=[("unit_spend", "desc"), ("customer_id", "desc")],
            agg_func="mean",
            ignore_ties=False,
        )

        expected_columns = [
            "unit_spend_rank",
            "customer_id_rank",
            "composite_rank",
        ]

        for col in expected_columns:
            assert col in cr.df.columns
