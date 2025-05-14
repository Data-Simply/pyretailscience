"""Integration tests for Cohort Analysis with BigQuery."""

import pandas as pd
import pytest

from pyretailscience.analysis.cohort import CohortAnalysis


class TestCohortAnalysisBigQuery:
    """Integration tests for Cohort Analysis using real BigQuery data."""

    def test_cohort_computation_bigquery(self, transactions_table):
        """Tests cohort computation logic using BigQuery data."""
        cohort = CohortAnalysis(
            df=transactions_table,
            aggregation_column="unit_spend",
            agg_func="nunique",
            period="month",
            percentage=False,
        )
        result = cohort.table
        assert not result.empty, "Cohort table should not be empty for valid BigQuery data"
        assert isinstance(result, pd.DataFrame)

    def test_invalid_period(self, transactions_table):
        """Test if an invalid period raises an error."""
        invalid_period = "m"
        with pytest.raises(
            ValueError,
            match=f"Invalid period '{invalid_period}'. Allowed values: {CohortAnalysis.VALID_PERIODS}",
        ):
            CohortAnalysis(
                df=transactions_table,
                aggregation_column="unit_spend",
                period=invalid_period,
            )

    def test_cohort_percentage(self, transactions_table):
        """Tests cohort analysis with percentage=True."""
        cohort = CohortAnalysis(
            df=transactions_table,
            aggregation_column="unit_spend",
            agg_func="sum",
            period="month",
            percentage=True,
        )
        result = cohort.table
        assert not result.empty
        assert result.max().max() <= 1.0, "Values should be <= 1 when percentage=True"
