import pandas as pd
import pytest

from pyretailscience.gain_loss import GainLoss


@pytest.fixture
def sample_data() -> pd.DataFrame:
    # fmt: off
    df = pd.DataFrame(
        {
            "group_id": [
                1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
            ],
            "customer_id": [
                2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 5, 6, 7, 7, 8, 9, 10, 1, 2, 4, 5, 6, 6, 7, 8, 9, 10, 6, 7, 7, 8, 9, 10
            ],

            "time_period": [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            ],
            "category_0_name": [
                "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b",
                "b", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b",
            ],
            "total_price": [
                1.0, 6.0, 4.0, 10.0, 5.0, 5.0, 20.0, 10.0, 5.0, 16.0, 5.0, 4.0, 3.0, 2.0, 5.0, 5.0, 10.0,
                10.0, 10.0, 1.0, 10.0, 3.0, 7.0, 5.0, 5.0, 12.0, 10.0, 1.0, 13.0, 7.0, 9.0, 3.0, 12.0
            ],
        }
    )
    # fmt: on
    return df


def test_calc_gain_loss(sample_data):
    gl_df = GainLoss._calc_gain_loss(
        df=sample_data,
        p1_index=sample_data["time_period"] == 1,
        p2_index=sample_data["time_period"] == 2,
        focus_group_index=sample_data["category_0_name"] == "a",
        comparison_group_index=sample_data["category_0_name"] == "b",
        value_col="total_price",
    )

    ret_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "focus_p1": [0.0, 1.0, 10.0, 10.0, 5.0, 5.0, 20.0, 10.0, 5.0, 16.0],
            "comparison_p1": [0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 5.0, 5.0, 5.0, 10.0],
            "total_p1": [0.0, 1.0, 10.0, 10.0, 10.0, 9.0, 25.0, 15.0, 10.0, 26.0],
            "focus_p2": [10.0, 10.0, 0.0, 1.0, 10.0, 10.0, 5.0, 5.0, 12.0, 10.0],
            "comparison_p2": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 9.0, 3.0, 12.0],
            "total_p2": [10.0, 10.0, 0.0, 1.0, 10.0, 11.0, 25.0, 14.0, 15.0, 22.0],
            "focus_diff": [10.0, 9.0, -10.0, -9.0, 5.0, 5.0, -15.0, -5.0, 7.0, -6.0],
            "comparison_diff": [0.0, 0.0, 0.0, 0.0, -5.0, -3.0, 15.0, 4.0, -2.0, 2.0],
            "total_diff": [10.0, 9.0, -10.0, -9.0, 0.0, 2.0, 0.0, -1.0, 5.0, -4.0],
            "switch_from_comparison": [0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 2.0, 0.0],
            "switch_to_comparison": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -15.0, -4.0, 0.0, -2.0],
            "new": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "lost": [0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "increased_focus": [0.0, 9.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0],
            "decreased_focus": [0.0, 0.0, 0.0, -9.0, 0.0, 0.0, 0.0, -1.0, 0.0, -4.0],
        }
    ).set_index("customer_id")

    assert gl_df.equals(ret_df)


def test_calc_gain_loss_groups(sample_data):
    gl_df = GainLoss._calc_gain_loss(
        df=sample_data,
        p1_index=sample_data["time_period"] == 1,
        p2_index=sample_data["time_period"] == 2,
        focus_group_index=sample_data["category_0_name"] == "a",
        comparison_group_index=sample_data["category_0_name"] == "b",
        value_col="total_price",
        group_col="group_id",
    )

    ret_df = pd.DataFrame(
        {
            "group_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "focus_p1": [0.0, 1.0, 10.0, 10.0, 5.0, 5.0, 20.0, 10.0, 5.0, 16.0],
            "comparison_p1": [0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 5.0, 5.0, 5.0, 10.0],
            "total_p1": [0.0, 1.0, 10.0, 10.0, 10.0, 9.0, 25.0, 15.0, 10.0, 26.0],
            "focus_p2": [10.0, 10.0, 0.0, 1.0, 10.0, 10.0, 5.0, 5.0, 12.0, 10.0],
            "comparison_p2": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 9.0, 3.0, 12.0],
            "total_p2": [10.0, 10.0, 0.0, 1.0, 10.0, 11.0, 25.0, 14.0, 15.0, 22.0],
            "focus_diff": [10.0, 9.0, -10.0, -9.0, 5.0, 5.0, -15.0, -5.0, 7.0, -6.0],
            "comparison_diff": [0.0, 0.0, 0.0, 0.0, -5.0, -3.0, 15.0, 4.0, -2.0, 2.0],
            "total_diff": [10.0, 9.0, -10.0, -9.0, 0.0, 2.0, 0.0, -1.0, 5.0, -4.0],
            "switch_from_comparison": [0.0, 0.0, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 2.0, 0.0],
            "switch_to_comparison": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -15.0, -4.0, 0.0, -2.0],
            "new": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "lost": [0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "increased_focus": [0.0, 9.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0],
            "decreased_focus": [0.0, 0.0, 0.0, -9.0, 0.0, 0.0, 0.0, -1.0, 0.0, -4.0],
        }
    )
    ret_df["customer_id"] = ret_df["customer_id"].astype("category")
    ret_df = ret_df.set_index(["group_id", "customer_id"])

    assert gl_df.equals(ret_df)


def test_calc_gain_loss_nunique():
    # fmt: off
    df = pd.DataFrame(
        {
            "customer_id": [2, 3, 3, 4, 6, 7, 9, 10, 5, 6, 6, 9, 10, 1, 2, 4, 5, 5, 8, 9, 10, 7, 7, 8, 9, 10],
            "time_period": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "category_0_name": ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b',
                                'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
        }
    )
    # fmt: on

    gl_df = GainLoss._calc_gain_loss(
        df=df,
        p1_index=df["time_period"] == 1,
        p2_index=df["time_period"] == 2,
        focus_group_index=df["category_0_name"] == "a",
        comparison_group_index=df["category_0_name"] == "b",
        value_col="customer_id",
        agg_func="nunique",
    )

    ret_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "focus_p1": [0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            "comparison_p1": [0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            "total_p1": [0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            "focus_p2": [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
            "comparison_p2": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            "total_p2": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            "focus_diff": [1, 0, -1, 0, 1, -1, -1, 1, 0, 0],
            "comparison_diff": [0, 0, 0, 0, -1, -1, 1, 1, 0, 0],
            "total_diff": [1, 0, -1, 0, 0, -1, 0, 1, 0, 0],
            "switch_from_comparison": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "switch_to_comparison": [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            "new": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "lost": [0, 0, -1, 0, 0, -1, 0, 0, 0, 0],
            "increased_focus": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "decreased_focus": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    ret_df["customer_id"] = ret_df["customer_id"].astype("category")
    ret_df = ret_df.set_index(["customer_id"])

    assert gl_df.equals(ret_df)
