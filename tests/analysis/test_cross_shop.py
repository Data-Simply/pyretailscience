"""Tests for the Cross Shop module."""

import pandas as pd
import pytest

from pyretailscience.analysis.cross_shop import CrossShop
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10],
            "category_1_name": [
                "Jeans",
                "Shoes",
                "Dresses",
                "Hats",
                "Shoes",
                "Jeans",
                "Jeans",
                "Shoes",
                "Dresses",
                "Jeans",
                "Shoes",
                "Jeans",
            ],
            cols.unit_spend: [10, 20, 30, 40, 20, 50, 10, 20, 30, 15, 40, 50],
        },
    )


def test_calc_cross_shop_two_groups(sample_data):
    """Test the _calc_cross_shop method with two groups."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
    )
    ret_df = pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group_1": pd.Series([1, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype="int32"),
            "group_2": pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype="int32"),
            "groups": [(1, 0), (0, 1), (0, 0), (0, 0), (1, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)],
            cols.unit_spend: [10, 20, 30, 40, 70, 10, 20, 45, 40, 50],
        },
    ).set_index(cols.customer_id)

    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_three_groups(sample_data):
    """Test the _calc_cross_shop method with three groups."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
    )
    ret_df = pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "group_1": pd.Series([1, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype="int32"),
            "group_2": pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype="int32"),
            "group_3": pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype="int32"),
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
            cols.unit_spend: [10, 20, 30, 40, 70, 10, 20, 45, 40, 50],
        },
    ).set_index(cols.customer_id)

    pd.testing.assert_frame_equal(cross_shop_df, ret_df, check_dtype=False)


def test_calc_cross_shop_three_groups_customer_id_nunique(sample_data):
    """Test the _calc_cross_shop method with three groups and customer_id as the value column."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.customer_id,
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
            cols.customer_id: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    ret_df.index.name = cols.customer_id
    ret_df = ret_df.astype({"group_1": "int32", "group_2": "int32", "group_3": "int32"})
    assert cross_shop_df.equals(ret_df)


def test_calc_cross_shop_table(sample_data):
    """Test the _calc_cross_shop_table method."""
    cross_shop_df = CrossShop._calc_cross_shop(
        sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.unit_spend,
    )
    cross_shop_table = CrossShop._calc_cross_shop_table(
        cross_shop_df,
        value_col=cols.unit_spend,
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
            cols.unit_spend: [40, 30, 80, 70, 45, 70],
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
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.customer_id,
        agg_func="nunique",
    )
    cross_shop_table = CrossShop._calc_cross_shop_table(
        cross_shop_df,
        value_col=cols.customer_id,
    )
    ret_df = pd.DataFrame(
        {
            "groups": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)],
            cols.customer_id: [1, 1, 3, 3, 1, 1],
            "percent": [0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
        },
    )

    assert cross_shop_table.equals(ret_df)


def test_calc_cross_shop_invalid_group_3(sample_data):
    """Test that _calc_cross_shop raises ValueError if only one of group_3_col or group_3_val is provided."""
    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated, then the other must be as well"):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
        )

    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated, then the other must be as well"):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_val="T-Shirts",
        )
