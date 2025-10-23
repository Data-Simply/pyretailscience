"""Tests for Shopping Missions discovery module.

This test file contains shared fixtures and utilities for testing both
SPPMI and TF-IDF implementations of shopping missions discovery.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Generate realistic shopping mission transaction data.

    Creates transactions representing different shopping missions:
    - Quick lunch (sandwich, drink, chips)
    - Weekly grocery (milk, bread, eggs, vegetables)
    - Breakfast (cereal, milk, bananas)
    - Evening snack (ice cream, cookies, soda)
    - Party prep (beer, chips, salsa, cups)

    Returns:
        pd.DataFrame: Transaction data with transaction_id and item columns.
    """
    # Define mission templates (items commonly bought together)
    quick_lunch = ["sandwich", "drink", "chips", "fruit"]
    weekly_grocery = ["milk", "bread", "eggs", "chicken", "vegetables", "cheese"]
    breakfast = ["cereal", "milk", "bananas", "yogurt", "orange_juice"]
    evening_snack = ["ice_cream", "cookies", "soda", "candy"]
    party_prep = ["beer", "chips", "salsa", "cups", "napkins", "guacamole"]

    missions = [quick_lunch, weekly_grocery, breakfast, evening_snack, party_prep]

    transactions = []
    transaction_id = 1

    # Generate transactions for each mission type
    rng = np.random.default_rng(42)
    for mission in missions:
        # Create 20 transactions per mission type
        for _ in range(20):
            # Select 2-5 items from the mission randomly
            n_items = rng.integers(2, min(6, len(mission) + 1))
            items = rng.choice(mission, size=n_items, replace=False)

            transactions.extend([{"transaction_id": transaction_id, "item": item} for item in items])

            transaction_id += 1

    # Add some noise transactions (random combinations)
    for _ in range(10):
        # Mix items from different missions
        all_items = [item for mission in missions for item in mission]
        n_items = rng.integers(2, 4)
        items = rng.choice(all_items, size=n_items, replace=False)

        transactions.extend([{"transaction_id": transaction_id, "item": item} for item in items])

        transaction_id += 1

    return pd.DataFrame(transactions)


@pytest.fixture
def minimal_transactions() -> pd.DataFrame:
    """Generate minimal transaction data for edge case testing.

    Returns:
        pd.DataFrame: Small transaction dataset.
    """
    return pd.DataFrame(
        {
            "transaction_id": [1, 1, 2, 2, 3, 3, 3],
            "item": ["bread", "milk", "bread", "butter", "milk", "eggs", "bread"],
        },
    )


@pytest.fixture
def single_item_transactions() -> pd.DataFrame:
    """Generate transactions with single items only.

    Returns:
        pd.DataFrame: Transactions where each has only one item.
    """
    return pd.DataFrame({"transaction_id": [1, 2, 3, 4, 5], "item": ["bread", "milk", "eggs", "bread", "milk"]})


@pytest.fixture
def expected_mission_profiles() -> dict[str, list[str]]:
    """Expected top items for each mission type.

    Returns:
        dict: Mapping of mission descriptions to expected top items.
    """
    return {
        "quick_lunch": ["sandwich", "drink", "chips", "fruit"],
        "weekly_grocery": ["milk", "bread", "eggs", "chicken", "vegetables", "cheese"],
        "breakfast": ["cereal", "milk", "bananas", "yogurt", "orange_juice"],
        "evening_snack": ["ice_cream", "cookies", "soda", "candy"],
        "party_prep": ["beer", "chips", "salsa", "cups", "napkins", "guacamole"],
    }


class TestShoppingMissionsBase:
    """Base test class for shopping missions implementations.

    Subclasses will test specific implementations (SPPMI vs TF-IDF).
    """

    def test_accepts_pandas_dataframe(self, sample_transactions):
        """Test that the class accepts pandas DataFrame input."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")

    def test_accepts_ibis_table(self, sample_transactions):
        """Test that the class accepts ibis Table input."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")

    def test_raises_on_missing_columns(self):
        """Test that appropriate error is raised for missing columns."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")

    def test_minimum_items_per_transaction_filter(self, sample_transactions):
        """Test filtering of transactions with too few items."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")

    def test_mission_assignment_completeness(self, sample_transactions):
        """Test that all transactions are assigned to missions."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")

    def test_mission_profiles_have_top_items(self, sample_transactions):
        """Test that mission profiles return top items."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")

    def test_mission_summary_statistics(self, sample_transactions):
        """Test that mission summary returns expected statistics."""
        # To be implemented by subclasses
        pytest.skip("To be implemented in subclass")
