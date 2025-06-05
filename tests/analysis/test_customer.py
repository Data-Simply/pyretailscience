"""Tests for the customer class - focusing on column name overrides."""

import datetime

import pandas as pd
import pytest

from pyretailscience.analysis.customer import DaysBetweenPurchases, PurchasesPerCustomer, TransactionChurn
from pyretailscience.options import option_context


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """Returns a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "transaction_id": list(range(12)),
            "customer_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4],
            "unit_spend": [3.23, 3.35, 6.00, 4.50, 5.10, 7.20, 3.80, 4.90, 6.50, 2.10, 8.00, 3.50],
            "transaction_date": [
                datetime.date(2023, 1, 15),
                datetime.date(2023, 1, 20),
                datetime.date(2023, 2, 5),
                datetime.date(2023, 2, 10),
                datetime.date(2023, 3, 1),
                datetime.date(2023, 3, 15),
                datetime.date(2023, 3, 20),
                datetime.date(2023, 4, 10),
                datetime.date(2023, 4, 25),
                datetime.date(2023, 5, 5),
                datetime.date(2023, 5, 20),
                datetime.date(2023, 6, 10),
            ],
        },
    )


class TestPurchasesPerCustomer:
    """Test PurchasesPerCustomer functionality."""

    def test_with_custom_column_names(self, transactions_df):
        """Test PurchasesPerCustomer with custom column names."""
        custom_transactions_df = transactions_df.rename(
            columns={
                "transaction_id": "txn_id",
                "customer_id": "cust_id",
            },
        )

        with option_context("column.customer_id", "cust_id", "column.transaction_id", "txn_id"):
            ppc = PurchasesPerCustomer(df=custom_transactions_df)

            assert hasattr(ppc, "cust_purchases_s"), "Should have cust_purchases_s attribute"
            assert isinstance(ppc.cust_purchases_s, pd.Series), "Should be a pandas Series"
            assert not ppc.cust_purchases_s.empty, "Should not be empty"


class TestDaysBetweenPurchases:
    """Test DaysBetweenPurchases functionality."""

    def test_with_custom_column_names(self, transactions_df):
        """Test DaysBetweenPurchases with custom column names."""
        custom_transactions_df = transactions_df.rename(
            columns={
                "customer_id": "cust_id",
                "transaction_date": "txn_date",
            },
        )
        custom_transactions_df["txn_date"] = pd.to_datetime(custom_transactions_df["txn_date"])

        with option_context("column.customer_id", "cust_id", "column.transaction_date", "txn_date"):
            dbp = DaysBetweenPurchases(df=custom_transactions_df)

            assert hasattr(dbp, "purchase_dist_s"), "Should have purchase_dist_s attribute"
            assert isinstance(dbp.purchase_dist_s, pd.Series), "Should be a pandas Series"
            assert not dbp.purchase_dist_s.empty, "Should not be empty"


class TestTransactionChurn:
    """Test TransactionChurn functionality."""

    def test_with_custom_column_names(self, transactions_df):
        """Test TransactionChurn with custom column names."""
        custom_transactions_df = transactions_df.rename(
            columns={
                "customer_id": "cust_id",
                "transaction_date": "txn_date",
            },
        )
        custom_transactions_df["txn_date"] = pd.to_datetime(custom_transactions_df["txn_date"])

        with option_context("column.customer_id", "cust_id", "column.transaction_date", "txn_date"):
            tc = TransactionChurn(df=custom_transactions_df, churn_period=30)

            assert hasattr(tc, "purchase_dist_df"), "Should have purchase_dist_df attribute"
            assert hasattr(tc, "n_unique_customers"), "Should have n_unique_customers attribute"
            assert isinstance(tc.purchase_dist_df, pd.DataFrame), "Should be a pandas DataFrame"
            assert isinstance(tc.n_unique_customers, int), "Should be an integer"
