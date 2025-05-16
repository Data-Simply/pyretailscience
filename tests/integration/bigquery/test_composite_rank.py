"""Integration tests for Composite Rank Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.composite_rank import CompositeRank


@pytest.fixture
def test_transactions_table(transactions_table):
    """Fetch test transactions data from BigQuery and convert it to a pandas DataFrame.

    The expected table should include columns like `product_id`, `unit_spend`, and `customer_id`.
    Adds a calculated column `spend_per_customer`.
    """
    try:
        ibis_table = transactions_table.mutate(
            spend_per_customer=transactions_table.unit_spend / transactions_table.customer_id,
        )

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to fetch or preprocess test data: {e}")
    else:
        return ibis_table


def test_composite_rank_basic(test_transactions_table):
    """Test basic CompositeRank functionality with BigQuery data."""
    rank_cols = [
        ("unit_spend", "desc"),
        ("customer_id", "desc"),
        ("spend_per_customer", "desc"),
    ]
    try:
        result = CompositeRank(
            df=test_transactions_table,
            rank_cols=rank_cols,
            agg_func="mean",
            ignore_ties=False,
        )
        assert result is not None
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"CompositeRank basic test failed: {e}")


@pytest.mark.parametrize("ignore_ties", [False, True])
def test_tie_handling(test_transactions_table, ignore_ties):
    """Test handling of ties during rank calculation."""
    rank_cols = [("unit_spend", "desc")]
    try:
        result = CompositeRank(
            df=test_transactions_table,
            rank_cols=rank_cols,
            agg_func="mean",
            ignore_ties=ignore_ties,
        )
        assert result is not None
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"CompositeRank failed with ignore_ties={ignore_ties}: {e}")
