"""Tests for the Cohort Analysis module."""

import datetime

import ibis
import pandas as pd
import pandas.testing as pdt
import pytest
from matplotlib import pyplot as plt

from pyretailscience.analysis.cohort import CohortPlot


class TestCohortAnalysis:
    """Tests for the Cohort Analysis module."""

    @pytest.fixture
    def transactions_df(self) -> pd.DataFrame:
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
    def expected_results_df(self) -> pd.DataFrame:
        """Expected cohort result DataFrame for comparison."""
        expected_df = pd.DataFrame(
            {"min_period_shopped": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-05-01"])},
        ).set_index("min_period_shopped")

        expected_df[[0, 1, 2, 3]] = [[2.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 0.0], [2.0, 1.0, 0.0, 0.0]]
        expected_df.columns.name = "period_since"

        return expected_df

    def test_cohort_computation(self, transactions_df, expected_results_df):
        """Tests cohort computation logic and compares output with expected DataFrame."""
        cohort = CohortPlot(
            df=transactions_df,
            customer_col="customer_id",
            date_col="transaction_date",
            aggregation_func="nunique",
            start_date=datetime.date(2023, 1, 1),
            x_period="month",
            y_period="month",
            percentage=False,
        )
        result = cohort.table
        expected_results_df.index = result.index.astype("datetime64[s]")
        pdt.assert_frame_equal(result, expected_results_df)

    def test_missing_columns(self):
        """Test if missing columns raise an error."""
        df = pd.DataFrame({"customer_id": [1, 2, 3], "unit_spend": [10, 20, 30]})
        with pytest.raises(ValueError, match="Missing required columns"):
            CohortPlot(df=df, customer_col="customer_id", date_col="transaction_date")

    def test_invalid_period(self, transactions_df):
        """Test if an invalid period raises an error."""
        with pytest.raises(ValueError, match="x_period .* must be equal to y_period"):
            CohortPlot(
                df=transactions_df,
                customer_col="customer_id",
                date_col="transaction_date",
                x_period="year",
                y_period="month",
            )

    def test_cohort_date_filtering(self, transactions_df):
        """Test if the cohort correctly filters data based on start and end dates."""
        start_date = datetime.date(2023, 2, 1)
        end_date = datetime.date(2023, 4, 30)

        cohort = CohortPlot(
            df=ibis.memtable(transactions_df),
            customer_col="customer_id",
            date_col="transaction_date",
            aggregation_func="nunique",
            start_date=start_date,
            end_date=end_date,
            x_period="month",
            y_period="month",
            percentage=True,
        )
        result = cohort.table
        assert result.index.min().date() >= start_date
        assert result.index.max().date() <= end_date

    def test_missing_start_date(self, transactions_df):
        """Test if missing start date column raises an error."""
        with pytest.raises(ValueError, match="start_date must be specified."):
            CohortPlot(df=transactions_df, customer_col="customer_id", date_col="transaction_date")

    @pytest.fixture
    def cohort_plot(self, transactions_df):
        """Returns a CohortPlot instance initialized with sample transaction data."""
        return CohortPlot(
            df=transactions_df,
            customer_col="customer_id",
            date_col="transaction_date",
            aggregation_func="nunique",
            start_date=datetime.date(2023, 1, 1),
            x_period="month",
            y_period="month",
            percentage=False,
        )

    def test_plot_basic(self, cohort_plot, transactions_df):
        """Tests basic plotting functionality using CohortPlot with default settings."""
        fig, ax = plt.subplots()
        cohort_plot.plot(
            df=transactions_df,
            x_col="transaction_date",
            group_col="customer_id",
            value_col="unit_spend",
            source_text="Source: Test Data",
            ax=ax,
        )
        assert ax is not None

    def test_plot_with_custom_ax(self, cohort_plot, transactions_df):
        """Tests if a custom title is correctly applied in the CohortPlot visualization."""
        fig, ax = plt.subplots()
        cohort_plot.plot(
            df=transactions_df,
            x_col="transaction_date",
            group_col="customer_id",
            value_col="unit_spend",
            title="Custom Cohort Plot",
            ax=ax,
        )
        assert ax.title.get_text() == "Custom Cohort Plot"
