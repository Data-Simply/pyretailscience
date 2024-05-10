import pandas as pd
from pyretailscience.segmentation import get_index


def test_get_index():
    # Test case: grp_cols only one column
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        }
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [77.77777778, 100, 106.0606]})
    output = get_index(
        df=df,
        grp_cols=["group_col"],
        filter_index=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)

    # Test case: grp_cols two columns
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
            "group_col2": ["D", "D", "D", "D", "D", "D", "E", "E", "E", "E", "E", "E"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    expected_output = pd.DataFrame(
        {
            "group_col2": ["D", "D", "D", "E", "E", "E"],
            "group_col1": ["A", "B", "C", "A", "B", "C"],
            "index": [77.77777778, 100, 106.0606, 98.51851852, 100, 100.9661836],
        }
    )
    output = get_index(
        df=df,
        grp_cols=["group_col2", "group_col1"],
        filter_index=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)

    # Test case: offset = 100
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        }
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [-22.22222222, 0, 6.060606061]})
    output = get_index(
        df=df,
        grp_cols=["group_col"],
        filter_index=df["filter_col"] == "X",
        value_col="value_col",
        offset=100,
    )
    pd.testing.assert_frame_equal(output, expected_output)

    # Test case: agg_func = "nunique"
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 5, 8],
        }
    )
    expected_output = pd.DataFrame(
        {
            "group_col1": ["A", "B", "C"],
            "index": [140, 140, 46.6666667],
        }
    )
    output = get_index(
        df=df,
        grp_cols=["group_col1"],
        filter_index=df["filter_col"] == "X",
        value_col="value_col",
        agg_func="nunique",
    )
    pd.testing.assert_frame_equal(output, expected_output)
