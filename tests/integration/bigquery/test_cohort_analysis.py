"""Integration tests for Cohort Analysis with BigQuery."""

from pyretailscience.analysis.cohort import CohortAnalysis


def test_cohort_analysis_with_bigquery(transactions_table):
    """Integration test for CohortAnalysis using BigQuery backend and Ibis table.

    This test ensures that the CohortAnalysis class initializes and executes successfully
    using BigQuery data with various combinations of aggregation parameters.
    """
    limited_table = transactions_table.limit(5000)

    CohortAnalysis(
        df=limited_table,
        aggregation_column="unit_spend",
        agg_func="sum",
        period="week",
        percentage=True,
    )
