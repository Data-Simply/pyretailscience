"""Integration tests for Cohort Analysis with PySpark."""

from pyretailscience.analysis.cohort import CohortAnalysis


def test_cohort_analysis_with_pyspark(transactions_table):
    """Integration test for CohortAnalysis using PySpark backend and Ibis table.

    This test ensures that the CohortAnalysis class initializes and executes successfully
    using PySpark data with various combinations of aggregation parameters.
    """
    limited_table = transactions_table.limit(5000)

    CohortAnalysis(
        df=limited_table,
        aggregation_column="unit_spend",
        agg_func="sum",
        period="week",
        percentage=True,
    )
