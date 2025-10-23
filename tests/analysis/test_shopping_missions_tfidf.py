"""Tests for TF-IDF-based Shopping Missions implementation."""

import ibis
import pandas as pd
import pytest

from pyretailscience.analysis.shopping_missions_tfidf import ShoppingMissionsTFIDF

# Import fixtures from base test file
from tests.analysis.test_shopping_missions import (
    minimal_transactions,
    sample_transactions,
    single_item_transactions,
)


class TestShoppingMissionsTFIDF:
    """Tests for ShoppingMissionsTFIDF implementation."""

    def test_accepts_pandas_dataframe(self, sample_transactions):
        """Test that the class accepts pandas DataFrame input."""
        missions = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
        )
        result = missions.df
        assert isinstance(result, pd.DataFrame)
        assert "mission" in result.columns

    def test_accepts_ibis_table(self, sample_transactions):
        """Test that the class accepts ibis Table input."""
        ibis_table = ibis.memtable(sample_transactions)
        missions = ShoppingMissionsTFIDF(
            ibis_table,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
        )
        result = missions.df
        assert isinstance(result, pd.DataFrame)
        assert "mission" in result.columns

    def test_raises_on_missing_columns(self):
        """Test that appropriate error is raised for missing columns."""
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="required but missing"):
            ShoppingMissionsTFIDF(
                df,
                transaction_col="transaction_id",
                item_col="item",
            )

    def test_minimum_items_per_transaction_filter(self, sample_transactions):
        """Test filtering of transactions with too few items."""
        # Add a single-item transaction
        single_item = pd.DataFrame({"transaction_id": [999], "item": ["lonely_item"]})
        df_with_single = pd.concat([sample_transactions, single_item], ignore_index=True)

        missions = ShoppingMissionsTFIDF(
            df_with_single,
            transaction_col="transaction_id",
            item_col="item",
            min_items_per_transaction=2,
            n_missions=5,
        )

        result = missions.df
        # Transaction 999 should be filtered out
        assert 999 not in result["transaction_id"].values

    def test_mission_assignment_completeness(self, sample_transactions):
        """Test that all transactions are assigned to missions."""
        missions = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
        )
        result = missions.df

        # All original transactions (after filtering) should have mission labels
        original_txns = sample_transactions.groupby("transaction_id").size()
        original_txns = original_txns[original_txns >= 2].index  # After min_items filter

        result_txns = result["transaction_id"].unique()

        # Every filtered transaction should appear in results
        for txn in original_txns:
            assert txn in result_txns

    def test_mission_profiles_have_top_items(self, sample_transactions):
        """Test that mission profiles return top items."""
        missions = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
        )

        # Get first mission ID
        mission_id = missions.df["mission"].iloc[0]
        profile = missions.get_mission_profile(mission_id, top_n=5)

        assert isinstance(profile, pd.DataFrame)
        assert "item" in profile.columns
        assert "frequency" in profile.columns
        assert "percentage" in profile.columns
        assert len(profile) <= 5

    def test_mission_summary_statistics(self, sample_transactions):
        """Test that mission summary returns expected statistics."""
        missions = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
        )

        summary = missions.get_mission_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "mission" in summary.columns
        assert "n_transactions" in summary.columns
        assert "avg_items_per_transaction" in summary.columns
        assert "unique_items" in summary.columns
        assert "top_item" in summary.columns

    def test_correct_number_of_missions(self, sample_transactions):
        """Test that KMeans creates exactly n_missions clusters."""
        n_missions = 5
        missions = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=n_missions,
        )

        result = missions.df
        unique_missions = result["mission"].nunique()

        # KMeans should create exactly n_missions clusters
        assert unique_missions == n_missions

    def test_different_n_missions(self, sample_transactions):
        """Test that different n_missions affects results."""
        missions_3 = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=3,
        )

        missions_7 = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=7,
        )

        result_3 = missions_3.df
        result_7 = missions_7.df

        # Different n_missions should give different number of clusters
        assert result_3["mission"].nunique() == 3
        assert result_7["mission"].nunique() == 7

    def test_deterministic_with_same_seed(self, sample_transactions):
        """Test that results are deterministic with same random_state."""
        missions_1 = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
            random_state=42,
        )

        missions_2 = ShoppingMissionsTFIDF(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=5,
            random_state=42,
        )

        result_1 = missions_1.df.sort_values("transaction_id").reset_index(drop=True)
        result_2 = missions_2.df.sort_values("transaction_id").reset_index(drop=True)

        # Results should be identical with same seed
        pd.testing.assert_frame_equal(result_1, result_2)

    def test_handles_small_datasets(self, minimal_transactions):
        """Test that algorithm handles small datasets gracefully."""
        # Request more missions than possible
        missions = ShoppingMissionsTFIDF(
            minimal_transactions,
            transaction_col="transaction_id",
            item_col="item",
            n_missions=10,  # More than available transactions
        )

        result = missions.df

        # Should work without error
        assert isinstance(result, pd.DataFrame)
        assert "mission" in result.columns
        # Number of clusters should be <= number of transactions
        assert result["mission"].nunique() <= result["transaction_id"].nunique()
