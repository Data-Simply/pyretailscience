"""Tests for the columns_utils module."""

import ibis
import pandas as pd

from pyretailscience.utils.columns import filter_and_label_by_condition


def test_filter_and_label_by_condition():
    """Test cases for filtering and labeling by condition."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": ["toys", "shoes", "toys", "books"],
        },
    )
    table = ibis.memtable(df)

    result = filter_and_label_by_condition(
        table,
        {
            "toys": table["category"] == "toys",
            "shoes": table["category"] == "shoes",
        },
    )

    expected_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["toys", "shoes", "toys"],
            "label": ["toys", "shoes", "toys"],
        },
    )

    assert (
        result.execute()
        .sort_values("id")
        .reset_index(drop=True)
        .equals(
            expected_df.sort_values("id").reset_index(drop=True),
        )
    )
