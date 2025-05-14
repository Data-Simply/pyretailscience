"""Integration tests for Cohort Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.cohort import CohortAnalysis


@pytest.mark.parametrize(
    ("aggregation_column", "agg_func", "period", "percentage"),
    [
        ("customer_id", "nunique", "month", False),
        ("unit_spend", "sum", "week", True),
        ("unit_quantity", "mean", "quarter", False),
    ],
)
def test_cohort_analysis_with_bigquery(
    transactions_table,
    aggregation_column,
    agg_func,
    period,
    percentage,
):
    """Integration test for CohortAnalysis using BigQuery backend and Ibis table.

    This test ensures that the CohortAnalysis class initializes and executes successfully
    using BigQuery data with various combinations of aggregation parameters.
    """
    limited_table = transactions_table.limit(5000)

    try:
        CohortAnalysis(
            df=limited_table,
            aggregation_column=aggregation_column,
            agg_func=agg_func,
            period=period,
            percentage=percentage,
        )
    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"CohortAnalysis failed with aggregation_column={aggregation_column}, "
            f"agg_func={agg_func}, period={period}, percentage={percentage}: {e}",
        )
