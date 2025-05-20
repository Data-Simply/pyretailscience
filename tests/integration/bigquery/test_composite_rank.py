"""Integration tests for Composite Rank Analysis with BigQuery."""

import pytest

from pyretailscience.analysis.composite_rank import CompositeRank


@pytest.mark.parametrize("ignore_ties", [False, True])
def test_tie_handling(transactions_table, ignore_ties):
    """Test handling of ties during rank calculation."""
    rank_cols = [
        ("unit_spend", "desc"),
        ("customer_id", "desc"),
    ]
    result = CompositeRank(
        df=transactions_table,
        rank_cols=rank_cols,
        agg_func="mean",
        ignore_ties=ignore_ties,
    )
    assert result is not None
    executed_result = result.df
    assert executed_result is not None
