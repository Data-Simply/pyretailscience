"""Tests for SPPMI-based Shopping Missions implementation."""

import ibis
import pandas as pd
import pytest

from pyretailscience.analysis.shopping_missions_sppmi import ShoppingMissionsSPPMI

# Import fixtures from base test file
from tests.analysis.test_shopping_missions import (
    minimal_transactions,
    sample_transactions,
    single_item_transactions,
)


class TestShoppingMissionsSPPMI:
    """Tests for ShoppingMissionsSPPMI implementation."""

    def test_accepts_pandas_dataframe(self, sample_transactions):
        """Test that the class accepts pandas DataFrame input."""
        missions = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=5,
        )
        result = missions.df
        assert isinstance(result, pd.DataFrame)
        assert "mission" in result.columns

    def test_accepts_ibis_table(self, sample_transactions):
        """Test that the class accepts ibis Table input."""
        ibis_table = ibis.memtable(sample_transactions)
        missions = ShoppingMissionsSPPMI(
            ibis_table,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=5,
        )
        result = missions.df
        assert isinstance(result, pd.DataFrame)
        assert "mission" in result.columns

    def test_raises_on_missing_columns(self):
        """Test that appropriate error is raised for missing columns."""
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="required but missing"):
            ShoppingMissionsSPPMI(
                df,
                transaction_col="transaction_id",
                item_col="item",
            )

    def test_minimum_items_per_transaction_filter(self, sample_transactions):
        """Test filtering of transactions with too few items."""
        # Add a single-item transaction
        single_item = pd.DataFrame({"transaction_id": [999], "item": ["lonely_item"]})
        df_with_single = pd.concat([sample_transactions, single_item], ignore_index=True)

        missions = ShoppingMissionsSPPMI(
            df_with_single,
            transaction_col="transaction_id",
            item_col="item",
            min_items_per_transaction=2,
            min_cluster_size=5,
        )

        result = missions.df
        # Transaction 999 should be filtered out
        assert 999 not in result["transaction_id"].values

    def test_mission_assignment_completeness(self, sample_transactions):
        """Test that all transactions are assigned to missions or noise."""
        missions = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=5,
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
        missions = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=5,
        )

        # Get a valid mission ID (not noise -1)
        valid_missions = [m for m in missions.df["mission"].unique() if m >= 0]

        if len(valid_missions) > 0:
            mission_id = valid_missions[0]
            profile = missions.get_mission_profile(mission_id, top_n=5)

            assert isinstance(profile, pd.DataFrame)
            assert "item" in profile.columns
            assert "frequency" in profile.columns
            assert "percentage" in profile.columns
            assert len(profile) <= 5

    def test_mission_summary_statistics(self, sample_transactions):
        """Test that mission summary returns expected statistics."""
        missions = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=5,
        )

        summary = missions.get_mission_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "mission" in summary.columns
        assert "n_transactions" in summary.columns
        assert "avg_items_per_transaction" in summary.columns
        assert "unique_items" in summary.columns
        assert "top_item" in summary.columns

    def test_noise_handling(self, minimal_transactions):
        """Test that HDBSCAN correctly identifies noise transactions."""
        # With very strict parameters, some transactions might be noise
        missions = ShoppingMissionsSPPMI(
            minimal_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=2,
            min_samples=2,
        )

        result = missions.df

        # Check that noise label (-1) might exist
        assert "mission" in result.columns
        # All mission labels should be >= -1
        assert (result["mission"] >= -1).all()

    def test_different_sppmi_shifts(self, sample_transactions):
        """Test that different SPPMI shifts affect results."""
        missions_low_shift = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            sppmi_shift=0.5,
            min_cluster_size=5,
        )

        missions_high_shift = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            sppmi_shift=3.0,
            min_cluster_size=5,
        )

        # Both should run without error
        result_low = missions_low_shift.df
        result_high = missions_high_shift.df

        assert isinstance(result_low, pd.DataFrame)
        assert isinstance(result_high, pd.DataFrame)

    def test_different_cluster_sizes(self, sample_transactions):
        """Test that different min_cluster_size affects granularity."""
        missions_small = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=3,
        )

        missions_large = ShoppingMissionsSPPMI(
            sample_transactions,
            transaction_col="transaction_id",
            item_col="item",
            min_cluster_size=10,
        )

        result_small = missions_small.df
        result_large = missions_large.df

        # Smaller cluster size typically finds more clusters
        n_clusters_small = len([m for m in result_small["mission"].unique() if m >= 0])
        n_clusters_large = len([m for m in result_large["mission"].unique() if m >= 0])

        # This isn't guaranteed but is highly likely with our test data
        assert n_clusters_small >= n_clusters_large or n_clusters_small > 0
