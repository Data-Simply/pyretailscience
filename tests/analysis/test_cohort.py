"""Tests for the Cohort Analysis module."""

import datetime

import pandas as pd
import pandas.testing as pdt
import pytest

from pyretailscience.analysis.cohort import CohortAnalysis


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
            {
                0: [2.0, 1.0, 0.0, 0.0, 2.0],
                1: [1.0, 1.0, 0.0, 0.0, 1.0],
                2: [2.0, 1.0, 0.0, 0.0, 0.0],
                3: [1.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]),
        )

        expected_df.index.name = "min_period_shopped"
        expected_df.columns.name = "period_since"

        return expected_df

    def test_cohort_computation(self, transactions_df, expected_results_df):
        """Tests cohort computation logic and compares output with expected DataFrame."""
        cohort = CohortAnalysis(
            df=transactions_df,
            aggregation_column="unit_spend",
            agg_func="nunique",
            period="month",
            percentage=False,
        )
        result = cohort.table
        expected_results_df.index = result.index.astype("datetime64[ns]")
        pdt.assert_frame_equal(result, expected_results_df)

    def test_missing_columns(self):
        """Test if missing columns raise an error."""
        df = pd.DataFrame({"customer_id": [1, 2, 3], "unit_spend": [10, 20, 30]})
        with pytest.raises(ValueError, match="Missing required columns"):
            CohortAnalysis(
                df=df,
                aggregation_column="unit_spend",
            )

    def test_invalid_period(self, transactions_df):
        """Test if an invalid period raises an error."""
        invalid_period = "m"
        with pytest.raises(
            ValueError,
            match=f"Invalid period '{invalid_period}'. Allowed values: {CohortAnalysis.VALID_PERIODS}",
        ):
            CohortAnalysis(
                df=transactions_df,
                aggregation_column="unit_spend",
                period=invalid_period,
            )

    def test_cohort_percentage(self, transactions_df):
        """Tests cohort analysis with percentage=True."""
        cohort = CohortAnalysis(
            df=transactions_df,
            aggregation_column="unit_spend",
            agg_func="sum",
            period="month",
            percentage=True,
        )
        result = cohort.table
        assert (result.iloc[0] <= 1).all(), "Percentage values should be between 0 and 1"
