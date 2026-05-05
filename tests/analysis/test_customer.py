"""Tests for openretailscience.analysis.customer."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from openretailscience.analysis.customer import (
    DaysBetweenPurchases,
    PurchasesPerCustomer,
    TransactionChurn,
)
from openretailscience.options import option_context

# Expected derived values for the `transactions_df` fixture below. Per-customer purchase
# counts are [1, 2, 3, 4] and per-customer mean inter-purchase gaps are [14.0, 30.0, 31.0].
EXPECTED_UNIQUE_CUSTOMERS = 4
MEDIAN_PURCHASE_COUNT = 2.5
MEDIAN_DAYS_BETWEEN_PURCHASES = 30.0


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """A small retail fixture chosen so every derived metric has a known closed-form value.

    Per-customer purchase patterns:
    - 101: 3 transactions on 2024-01-01 / 01-31 / 03-01 — gaps of 30, 30 days (avg 30.0)
    - 102: 2 transactions on 2024-01-15 / 02-15 — single gap of 31 days (avg 31.0)
    - 103: 4 transactions on 2024-04-01 / 04-15 / 04-29 / 05-13 — gaps of 14, 14, 14 days (avg 14.0)
    - 104: 1 transaction on 2024-04-10 — excluded from any inter-purchase computation

    The fixture also yields predictable churn behaviour with churn_period=30 (boundary 2024-04-13):
    customers 101, 102, 104 fall before the boundary; customer 103 has only its first transaction
    inside the window.
    """
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "customer_id": [101, 101, 101, 102, 102, 103, 103, 103, 103, 104],
            "transaction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-31",
                    "2024-03-01",
                    "2024-01-15",
                    "2024-02-15",
                    "2024-04-01",
                    "2024-04-15",
                    "2024-04-29",
                    "2024-05-13",
                    "2024-04-10",
                ],
            ),
            "unit_spend": [50.0, 60.0, 70.0, 100.0, 110.0, 25.0, 30.0, 35.0, 40.0, 80.0],
        },
    )


@pytest.fixture
def expected_purchase_counts() -> pd.Series:
    """Per-customer transaction counts for the fixture, indexed by customer_id."""
    return pd.Series(
        [3, 2, 4, 1],
        index=pd.Index([101, 102, 103, 104], name="customer_id"),
        name="transaction_id",
    )


@pytest.fixture
def expected_days_between_purchases() -> pd.Series:
    """Per-customer mean inter-purchase gap in days (single-transaction customers are excluded)."""
    return pd.Series(
        [30.0, 31.0, 14.0],
        index=pd.Index([101, 102, 103], name="customer_id"),
        name="diff",
    )


@pytest.fixture
def expected_churn_table() -> pd.DataFrame:
    """Retained / churned counts and churned_pct per transaction_number for the fixture."""
    # Filtered transactions and their churn flags (churn boundary 2024-04-13):
    #   txn 1: cust 101, 102, 103 retained; cust 104 churned -> retained=3, churned=1, pct=0.25
    #   txn 2: cust 101 retained;            cust 102 churned -> retained=1, churned=1, pct=0.5
    #   txn 3: (no retained);                cust 101 churned -> retained=NaN, churned=1, pct=1.0
    return pd.DataFrame(
        {
            "retained": [3.0, 1.0, np.nan],
            "churned": [1.0, 1.0, 1.0],
            "churned_pct": [0.25, 0.5, 1.0],
        },
        index=pd.Index([1, 2, 3], name="transaction_number"),
    )


class TestPurchasesPerCustomer:
    """Behavioral tests for PurchasesPerCustomer."""

    def test_counts_unique_transactions_per_customer(self, transactions_df, expected_purchase_counts):
        """cust_purchases_s holds the unique transaction count per customer."""
        ppc = PurchasesPerCustomer(transactions_df)
        assert_series_equal(ppc.cust_purchases_s.sort_index(), expected_purchase_counts.sort_index())

    def test_purchases_percentile_returns_quantile_of_purchase_counts(self, transactions_df):
        """purchases_percentile reports the requested quantile of per-customer purchase counts."""
        ppc = PurchasesPerCustomer(transactions_df)
        # Counts sorted are [1, 2, 3, 4]; pandas linear interpolation puts the median at 2.5.
        assert ppc.purchases_percentile(0.5) == MEDIAN_PURCHASE_COUNT

    @pytest.mark.parametrize(
        ("threshold", "comparison", "expected"),
        [
            (2, "less_than_equal_to", 0.5),
            (3, "less_than", 0.5),
            (3, "equal_to", 0.25),
            (3, "greater_than", 0.25),
            (3, "greater_than_equal_to", 0.5),
            (3, "not_equal_to", 0.75),
            (5, "greater_than_equal_to", 0.0),
        ],
    )
    def test_find_purchase_percentile(self, transactions_df, threshold, comparison, expected):
        """find_purchase_percentile returns the share of customers matching the comparison."""
        ppc = PurchasesPerCustomer(transactions_df)
        assert ppc.find_purchase_percentile(threshold, comparison) == expected

    def test_find_purchase_percentile_invalid_comparison_raises(self, transactions_df):
        """Unknown comparison strings are rejected with a clear error."""
        ppc = PurchasesPerCustomer(transactions_df)
        with pytest.raises(ValueError, match="Comparison must be one of"):
            ppc.find_purchase_percentile(1, "foo")

    def test_missing_required_columns_raises(self):
        """Dropping a required column raises with the missing column listed."""
        df = pd.DataFrame({"customer_id": [1], "unit_spend": [1.0]})
        with pytest.raises(ValueError, match="transaction_id"):
            PurchasesPerCustomer(df)

    def test_with_custom_column_names(self, transactions_df, expected_purchase_counts):
        """Custom ColumnHelper names are honoured and produce identical counts."""
        renamed = transactions_df.rename(columns={"transaction_id": "txn_id", "customer_id": "cust_id"})
        with option_context("column.customer_id", "cust_id", "column.transaction_id", "txn_id"):
            ppc = PurchasesPerCustomer(df=renamed)

        expected = expected_purchase_counts.rename_axis("cust_id").rename("txn_id")
        assert_series_equal(ppc.cust_purchases_s.sort_index(), expected.sort_index())


class TestDaysBetweenPurchases:
    """Behavioral tests for DaysBetweenPurchases."""

    def test_computes_average_days_between_purchases(self, transactions_df, expected_days_between_purchases):
        """purchase_dist_s holds the per-customer mean inter-purchase gap, excluding single-transaction customers."""
        dbp = DaysBetweenPurchases(transactions_df)
        assert_series_equal(dbp.purchase_dist_s.sort_index(), expected_days_between_purchases.sort_index())

    def test_purchases_percentile_returns_quantile_of_gaps(self, transactions_df):
        """purchases_percentile reports the requested quantile of per-customer mean gaps."""
        dbp = DaysBetweenPurchases(transactions_df)
        # Sorted gaps are [14.0, 30.0, 31.0]; the median is 30.0.
        assert dbp.purchases_percentile(0.5) == MEDIAN_DAYS_BETWEEN_PURCHASES

    def test_missing_required_columns_raises(self):
        """Dropping a required column raises with the missing column listed."""
        df = pd.DataFrame({"customer_id": [1], "unit_spend": [1.0]})
        with pytest.raises(ValueError, match="transaction_date"):
            DaysBetweenPurchases(df)

    def test_with_custom_column_names(self, transactions_df, expected_days_between_purchases):
        """Custom ColumnHelper names are honoured and produce identical gaps."""
        renamed = transactions_df.rename(columns={"customer_id": "cust_id", "transaction_date": "txn_date"})
        with option_context("column.customer_id", "cust_id", "column.transaction_date", "txn_date"):
            dbp = DaysBetweenPurchases(df=renamed)

        expected = expected_days_between_purchases.rename_axis("cust_id")
        assert_series_equal(dbp.purchase_dist_s.sort_index(), expected.sort_index())


class TestTransactionChurn:
    """Behavioral tests for TransactionChurn."""

    def test_computes_retained_churned_and_rate_per_transaction_number(self, transactions_df, expected_churn_table):
        """purchase_dist_df reports retained, churned counts and the churned percentage."""
        tc = TransactionChurn(transactions_df, churn_period=30)
        assert_frame_equal(tc.purchase_dist_df.sort_index(), expected_churn_table)

    def test_counts_unique_customers_in_source(self, transactions_df):
        """n_unique_customers reflects the distinct customers in the input, not the filtered window."""
        tc = TransactionChurn(transactions_df, churn_period=30)
        assert tc.n_unique_customers == EXPECTED_UNIQUE_CUSTOMERS

    def test_missing_required_columns_raises(self):
        """Dropping a required column raises with the missing column listed."""
        df = pd.DataFrame({"customer_id": [1], "unit_spend": [1.0]})
        with pytest.raises(ValueError, match="transaction_date"):
            TransactionChurn(df, churn_period=30)

    def test_with_custom_column_names(self, transactions_df, expected_churn_table):
        """Custom ColumnHelper names are honoured and produce identical churn rates."""
        renamed = transactions_df.rename(columns={"customer_id": "cust_id", "transaction_date": "txn_date"})
        with option_context("column.customer_id", "cust_id", "column.transaction_date", "txn_date"):
            tc = TransactionChurn(df=renamed, churn_period=30)
        assert_frame_equal(tc.purchase_dist_df.sort_index(), expected_churn_table)
