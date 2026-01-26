"""Unified integration tests for Cohort Analysis with multiple database backends."""

from pyretailscience.analysis.cohort import CohortAnalysis


def test_cohort_analysis_integration(transactions_table):
    """Integration test for CohortAnalysis using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
    """
    limited_table = transactions_table.limit(5000)

    cohort_analysis = CohortAnalysis(
        df=limited_table,
        aggregation_column="unit_spend",
        agg_func="sum",
        period="week",
        percentage=True,
    )

    result = cohort_analysis.df
    assert result is not None
    assert len(result.columns) > 0
