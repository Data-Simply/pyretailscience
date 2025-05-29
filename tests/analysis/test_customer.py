"""Tests for the customer class."""

import datetime

import pandas as pd
import pytest
from matplotlib import pyplot as plt

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


@pytest.fixture
def custom_column_mappings() -> dict[str, str]:
    """Returns custom column mappings for testing."""
    return {
        "column.customer_id": "cust_id",
        "column.transaction_id": "txn_id",
        "column.transaction_date": "txn_date",
        "column.unit_spend": "spend_amount",
    }


@pytest.fixture
def custom_transactions_df(transactions_df, custom_column_mappings) -> pd.DataFrame:
    """Returns a DataFrame with custom column names."""
    rename_mapping = {
        "transaction_id": custom_column_mappings["column.transaction_id"],
        "customer_id": custom_column_mappings["column.customer_id"],
        "unit_spend": custom_column_mappings["column.unit_spend"],
        "transaction_date": custom_column_mappings["column.transaction_date"],
    }
    df = transactions_df.rename(columns=rename_mapping)
    df[custom_column_mappings["column.transaction_date"]] = pd.to_datetime(
        df[custom_column_mappings["column.transaction_date"]],
    )
    return df


class TestCustomerAnalysisBase:
    """Base class with common test utilities for customer analysis classes."""

    @staticmethod
    def assert_plot_functionality(analysis_instance, plot_configs: list):
        """Test plotting functionality with different configurations."""
        for config in plot_configs:
            fig, ax = plt.subplots()
            result_ax = analysis_instance.plot(ax=ax, **config)
            assert result_ax is not None, f"Plot with config {config} should return an axes object"
            plt.close(fig)


@pytest.mark.parametrize(
    ("analysis_class", "expected_attributes", "percentile_methods", "special_methods"),
    [
        (
            PurchasesPerCustomer,
            ["cust_purchases_s"],
            ["purchases_percentile"],
            [("find_purchase_percentile", (1, "greater_than_equal_to"), float)],
        ),
        (
            DaysBetweenPurchases,
            ["purchase_dist_s"],
            ["purchases_percentile"],
            [],
        ),
    ],
)
class TestCustomerAnalysisWithCustomColumns(TestCustomerAnalysisBase):
    """Parameterized tests for customer analysis classes with custom column names."""

    def test_initialization_and_basic_functionality(
        self,
        custom_transactions_df,
        custom_column_mappings,
        analysis_class: type[PurchasesPerCustomer | DaysBetweenPurchases],
        expected_attributes: list,
        percentile_methods: list,
        special_methods: list,
    ):
        """Test initialization and basic functionality of analysis classes."""
        required_columns = self._get_required_columns(analysis_class, custom_column_mappings)

        with option_context(*[item for pair in required_columns.items() for item in pair]):
            instance = analysis_class(df=custom_transactions_df)

            # Test expected attributes exist and are correct type
            for attr in expected_attributes:
                assert hasattr(instance, attr), f"Should have {attr} attribute"
                attr_value = getattr(instance, attr)
                assert isinstance(attr_value, pd.Series), f"{attr} should be a pandas Series"
                assert not attr_value.empty, f"{attr} should not be empty"
                assert (attr_value > 0).all(), f"All values in {attr} should be positive"

            # Test percentile methods
            for method_name in percentile_methods:
                method = getattr(instance, method_name)
                result = method(0.5)
                assert isinstance(result, float), f"{method_name} should return a float"
                assert result > 0, f"{method_name} result should be positive"

            # Test special methods
            for method_name, args, expected_type in special_methods:
                method = getattr(instance, method_name)
                result = method(*args)
                assert isinstance(result, expected_type), f"{method_name} should return {expected_type.__name__}"
                if expected_type == float:
                    assert 0 <= result <= 1, f"{method_name} percentile should be between 0 and 1"

    def test_plotting_functionality(
        self,
        custom_transactions_df,
        custom_column_mappings,
        analysis_class: type[PurchasesPerCustomer | DaysBetweenPurchases],
        expected_attributes: list,
        percentile_methods: list,
        special_methods: list,
    ):
        """Test plotting functionality with various configurations."""
        required_columns = self._get_required_columns(analysis_class, custom_column_mappings)

        plot_configs = [
            {"bins": 5},
            {"cumulative": True, "bins": 5},
            {"percentile_line": 0.8, "bins": 5}
            if analysis_class == PurchasesPerCustomer
            else {"percentile_line": 0.7, "bins": 5},
        ]

        with option_context(*[item for pair in required_columns.items() for item in pair]):
            instance = analysis_class(df=custom_transactions_df)
            self.assert_plot_functionality(instance, plot_configs)

    @staticmethod
    def _get_required_columns(analysis_class: type, custom_column_mappings: dict[str, str]) -> dict[str, str]:
        """Get required column mappings for each analysis class."""
        base_columns = {
            "column.customer_id": custom_column_mappings["column.customer_id"],
        }

        if analysis_class == PurchasesPerCustomer:
            base_columns["column.transaction_id"] = custom_column_mappings["column.transaction_id"]
        elif analysis_class == DaysBetweenPurchases:
            base_columns["column.transaction_date"] = custom_column_mappings["column.transaction_date"]

        return base_columns


class TestDaysBetweenPurchasesStaticMethod:
    """Test static methods specific to DaysBetweenPurchases."""

    def test_static_calculation_method(self, custom_transactions_df, custom_column_mappings):
        """Test the static calculation method directly."""
        required_columns = {
            "column.customer_id": custom_column_mappings["column.customer_id"],
            "column.transaction_date": custom_column_mappings["column.transaction_date"],
        }

        with option_context(*[item for pair in required_columns.items() for item in pair]):
            calculated_days = DaysBetweenPurchases._calculate_days_between_purchases(custom_transactions_df)
            assert isinstance(calculated_days, pd.Series), "Static method should return a pandas Series"
            assert not calculated_days.empty, "Calculated days should not be empty"
            assert (calculated_days > 0).all(), "All calculated days should be positive"


@pytest.mark.parametrize("churn_period", [7, 30, 60])
class TestTransactionChurnWithCustomColumns(TestCustomerAnalysisBase):
    """Test TransactionChurn class with different churn periods."""

    def test_initialization_and_data_integrity(self, custom_transactions_df, custom_column_mappings, churn_period):
        """Test TransactionChurn initialization and data integrity."""
        required_columns = {
            "column.customer_id": custom_column_mappings["column.customer_id"],
            "column.transaction_date": custom_column_mappings["column.transaction_date"],
        }

        with option_context(*[item for pair in required_columns.items() for item in pair]):
            tc = TransactionChurn(df=custom_transactions_df, churn_period=churn_period)

            assert hasattr(tc, "purchase_dist_df"), "Should have purchase_dist_df attribute"
            assert hasattr(tc, "n_unique_customers"), "Should have n_unique_customers attribute"
            assert isinstance(tc.purchase_dist_df, pd.DataFrame), "purchase_dist_df should be a pandas DataFrame"
            assert isinstance(tc.n_unique_customers, int), "n_unique_customers should be an integer"
            assert not tc.purchase_dist_df.empty, "purchase_dist_df should not be empty"
            assert tc.n_unique_customers > 0, "Should have at least one unique customer"

            expected_columns = ["retained", "churned_pct"]
            for col in expected_columns:
                assert col in tc.purchase_dist_df.columns, f"Should have {col} column"

            # Test data integrity
            self._assert_data_integrity(tc.purchase_dist_df)

    def test_plotting_functionality(self, custom_transactions_df, custom_column_mappings, churn_period):
        """Test TransactionChurn plotting functionality."""
        required_columns = {
            "column.customer_id": custom_column_mappings["column.customer_id"],
            "column.transaction_date": custom_column_mappings["column.transaction_date"],
        }

        plot_configs = [
            {},
            {"cumulative": True},
        ]

        with option_context(*[item for pair in required_columns.items() for item in pair]):
            tc = TransactionChurn(df=custom_transactions_df, churn_period=churn_period)
            self.assert_plot_functionality(tc, plot_configs)

    @staticmethod
    def _assert_data_integrity(purchase_dist_df: pd.DataFrame):
        """Assert data integrity for TransactionChurn results."""
        valid_churn_pct = purchase_dist_df["churned_pct"].dropna()
        if not valid_churn_pct.empty:
            assert (valid_churn_pct >= 0).all(), "Churn percentages should be non-negative"
            assert (valid_churn_pct <= 1).all(), "Churn percentages should not exceed 100%"

        assert (purchase_dist_df["retained"].fillna(0) >= 0).all(), "Retained counts should be non-negative"

        if "churned" in purchase_dist_df.columns:
            assert (purchase_dist_df["churned"].fillna(0) >= 0).all(), "Churned counts should be non-negative"
