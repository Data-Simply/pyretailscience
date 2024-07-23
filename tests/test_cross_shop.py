"""Tests for the Cross Shop module."""

import pandas as pd
import pytest

from pyretailscience.cross_shop import CrossShop


@pytest.fixture()
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10],
            "group_1_idx": [True, False, False, False, False, True, True, False, False, True, False, True],
            "group_2_idx": [False, True, False, False, True, False, False, True, False, False, True, False],
            "group_3_idx": [False, False, True, False, False, False, False, False, True, False, False, False],
            "total_price": [10, 20, 30, 40, 20, 50, 10, 20, 30, 15, 40, 50],
        },
    )


def test_calc_cross_shop_two_groups(sample_data):
    """Test the _calc_cross_shop method with two groups."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_idx=sample_data["group_1_idx"],
        group_2_idx=sample_data["group_2_idx"],
    )
    ret_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group_1": [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            "group_2": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "groups": [(1, 0), (0, 1), (0, 0), (0, 0), (1, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)],
            "total_price": [10, 20, 30, 40, 70, 10, 20, 45, 40, 50],
        },
    ).set_index("customer_id")

    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_three_groups(sample_data):
    """Test the _calc_cross_shop method with three groups."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_idx=sample_data["group_1_idx"],
        group_2_idx=sample_data["group_2_idx"],
        group_3_idx=sample_data["group_3_idx"],
    )
    ret_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group_1": [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            "group_2": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "group_3": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            "groups": [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 0, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
            ],
            "total_price": [10, 20, 30, 40, 70, 10, 20, 45, 40, 50],
        },
    ).set_index("customer_id")

    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_two_groups_overlap_error(sample_data):
    """Test the _calc_cross_shop method with two groups and overlapping group indices."""
    with pytest.raises(ValueError):
        CrossShop._calc_cross_shop(
            sample_data,
            # Pass the same group index for both groups
            group_1_idx=sample_data["group_1_idx"],
            group_2_idx=sample_data["group_1_idx"],
        )


def test_calc_cross_shop_three_groups_overlap_error(sample_data):
    """Test the _calc_cross_shop method with three groups and overlapping group indices."""
    with pytest.raises(ValueError):
        CrossShop._calc_cross_shop(
            sample_data,
            # Pass the same group index for groups 1 and 3
            group_1_idx=sample_data["group_1_idx"],
            group_2_idx=sample_data["group_2_idx"],
            group_3_idx=sample_data["group_1_idx"],
        )


def test_calc_cross_shop_three_groups_customer_id_nunique(sample_data):
    """Test the _calc_cross_shop method with three groups and customer_id as the value column."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_idx=sample_data["group_1_idx"],
        group_2_idx=sample_data["group_2_idx"],
        group_3_idx=sample_data["group_3_idx"],
        value_col="customer_id",
        agg_func="nunique",
    )
    ret_df = pd.DataFrame(
        {
            "group_1": [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
            "group_2": [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            "group_3": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            "groups": [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 0, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
            ],
            "customer_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    ret_df.index.name = "customer_id"

    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_table(sample_data):
    """Test the _calc_cross_shop_table method."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_idx=sample_data["group_1_idx"],
        group_2_idx=sample_data["group_2_idx"],
        group_3_idx=sample_data["group_3_idx"],
        value_col="total_price",
    )
    cross_shop_table = CrossShop._calc_cross_shop_table(
        cross_shop_df,
        value_col="total_price",
    )
    ret_df = pd.DataFrame(
        {
            "groups": [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
            ],
            "total_price": [40, 30, 80, 70, 45, 70],
            "percent": [0.119402985, 0.089552239, 0.23880597, 0.208955224, 0.134328358, 0.208955224],
        },
    )

    # Equals should be using allclose for float columns but it needs
    ret_df["percent"] = ret_df["percent"].round(6)
    cross_shop_table["percent"] = cross_shop_table["percent"].round(6)

    assert cross_shop_table.equals(ret_df)


def test_calc_cross_shop_table_customer_id_nunique(sample_data):
    """Test the _calc_cross_shop_table method with customer_id as the value column."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_idx=sample_data["group_1_idx"],
        group_2_idx=sample_data["group_2_idx"],
        group_3_idx=sample_data["group_3_idx"],
        value_col="customer_id",
        agg_func="nunique",
    )
    cross_shop_table = CrossShop._calc_cross_shop_table(
        cross_shop_df,
        value_col="customer_id",
    )
    ret_df = pd.DataFrame(
        {
            "groups": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)],
            "customer_id": [1, 1, 3, 3, 1, 1],
            "percent": [0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
        },
    )

    assert cross_shop_table.equals(ret_df)


def test_calc_cross_shop_all_groups_false(sample_data):
    """Test the _calc_cross_shop method with all group indices set to False."""
    with pytest.raises(ValueError):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_idx=[False] * len(sample_data),
            group_2_idx=[False] * len(sample_data),
        )

    with pytest.raises(ValueError):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_idx=[False] * len(sample_data),
            group_2_idx=[False] * len(sample_data),
            group_3_idx=[False] * len(sample_data),
        )
