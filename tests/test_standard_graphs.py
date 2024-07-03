"""Tests for the standard_graphs module."""

import pandas as pd
import pytest

from pyretailscience.standard_graphs import get_indexes


def test_get_indexes_single_column():
    """Test that the function works with a single column index."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [77.77777778, 100, 106.0606]})
    output = get_indexes(
        df=df,
        index_col="group_col",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_two_columns():
    """Test that the function works with two columns as the index."""
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
            "group_col2": ["D", "D", "D", "D", "D", "D", "E", "E", "E", "E", "E", "E"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col2": ["D", "D", "D", "E", "E", "E"],
            "group_col1": ["A", "B", "C", "A", "B", "C"],
            "index": [77.77777778, 100, 106.0606, 98.51851852, 100, 100.9661836],
        },
    )
    output = get_indexes(
        df=df,
        index_col="group_col1",
        index_subgroup_col="group_col2",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_offset():
    """Test that the function works with an offset parameter."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [-22.22222222, 0, 6.060606061]})
    output = get_indexes(
        df=df,
        index_col="group_col",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
        offset=100,
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_agg_func():
    """Test that the function works with the nunique agg_func parameter."""
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 5, 8],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col1": ["A", "B", "C"],
            "index": [140, 140, 46.6666667],
        },
    )
    output = get_indexes(
        df=df,
        index_col="group_col1",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
        agg_func="nunique",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_index_filter_all_same():
    """Test that the function raises a ValueError when all the values in the index filter are the same."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "X", "X", "X", "X", "X"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    # Assert a value error will be reaised
    with pytest.raises(ValueError):
        get_indexes(
            df=df,
            df_index_filter=[True, True, True, True, True, True],
            index_col="group_col",
            value_col="value_col",
        )

    with pytest.raises(ValueError):
        get_indexes(
            df=df,
            df_index_filter=[False, False, False, False, False, False],
            index_col="group_col",
            value_col="value_col",
        )
