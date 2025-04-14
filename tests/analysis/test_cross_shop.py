"""Tests for the Cross Shop module."""

from unittest.mock import MagicMock, patch

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


def test_missing_columns_raises(sample_data):
    """Test that ValueError is raised when required columns are missing in df."""
    bad_data = sample_data.drop(columns=[cols.unit_spend])
    with pytest.raises(ValueError, match="The following columns are required but missing:"):
        CrossShop(
            df=bad_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
        )


def test_labels_length_mismatch(sample_data):
    """Test that ValueError is raised when labels length does not match group count."""
    with pytest.raises(ValueError, match="The number of labels must be equal to the number of group indexes given"):
        CrossShop(
            df=sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
            group_3_val="Dresses",
            labels=["A", "B"],
        )


def test_group_3_only_one_side_provided(sample_data):
    """Test that ValueError is raised when only group_3_col or group_3_val is given."""
    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated"):
        CrossShop._calc_cross_shop(
            sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
        )


@patch("pyretailscience.plots.venn.plot")
def test_plot_passes_parameters_correctly(mock_venn_plot, sample_data):
    """Test that the plot method passes parameters correctly to venn.plot."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Jeans", "Shoes"],
    )

    title = "Test Title"
    source_text = "Source: Test Data"
    vary_size = True
    figsize = (10, 8)
    ax = MagicMock()

    def custom_formatter(x):
        return f"Custom {x}"

    cross_shop.plot(
        title=title,
        source_text=source_text,
        vary_size=vary_size,
        figsize=figsize,
        ax=ax,
        subset_label_formatter=custom_formatter,
        extra_param="test",
    )

    mock_venn_plot.assert_called_once()
    _, kwargs = mock_venn_plot.call_args

    assert kwargs["df"] is cross_shop.cross_shop_table_df
    assert kwargs["labels"] == ["Jeans", "Shoes"]
    assert kwargs["title"] == title
    assert kwargs["source_text"] == source_text
    assert kwargs["vary_size"] == vary_size
    assert kwargs["figsize"] == figsize
    assert kwargs["ax"] == ax
    assert kwargs["subset_label_formatter"] == custom_formatter
    assert kwargs["extra_param"] == "test"


def test_plot_with_default_labels(sample_data):
    """Test that plot generates default alphabetical labels when none are provided."""
    with patch("pyretailscience.plots.venn.plot") as mock_venn_plot:
        cross_shop = CrossShop(
            df=sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
        )

        cross_shop.plot()

        _, kwargs = mock_venn_plot.call_args
        assert kwargs["labels"] == ["A", "B"]


def test_plot_with_default_labels_three_groups(sample_data):
    """Test that plot generates default alphabetical labels for three groups."""
    with patch("pyretailscience.plots.venn.plot") as mock_venn_plot:
        cross_shop = CrossShop(
            df=sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
            group_3_val="Dresses",
        )

        cross_shop.plot()

        _, kwargs = mock_venn_plot.call_args
        assert kwargs["labels"] == ["A", "B", "C"]


def test_plot_returns_axes(sample_data):
    """Test the plot method returns matplotlib axes."""
    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Jeans", "Shoes"],
    )
    ax = cross_shop.plot(title="Test Title")
    assert hasattr(ax, "plot")


def test_cross_shop_with_non_empty_dataframe():
    """Test CrossShop initialization with a minimal non-empty DataFrame."""
    group_count = 2
    minimal_df = pd.DataFrame(
        {
            cols.customer_id: [1, 2],
            cols.unit_spend: [10, 20],
            "category": ["Jeans", "Shoes"],
        },
    )

    cross_shop = CrossShop(
        df=minimal_df,
        group_1_col="category",
        group_1_val="Jeans",
        group_2_col="category",
        group_2_val="Shoes",
    )

    assert cross_shop.group_count == group_count
    assert not cross_shop.cross_shop_df.empty
    assert not cross_shop.cross_shop_table_df.empty


def test_cross_shop_with_custom_value_col_and_agg_func(sample_data):
    """Test CrossShop initialization with custom value_col and agg_func."""
    customer_value = 70.0
    sample_data["custom_value"] = sample_data[cols.unit_spend] * 2

    cross_shop = CrossShop(
        df=sample_data,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        value_col="custom_value",
        agg_func="mean",
    )

    assert "custom_value" in cross_shop.cross_shop_df.columns

    assert cross_shop.cross_shop_df.loc[5, "custom_value"] == customer_value


def test_plot_with_additional_kwargs(sample_data):
    """Test plot method passing through additional kwargs to venn.plot."""
    with patch("pyretailscience.plots.venn.plot") as mock_venn_plot:
        cross_shop = CrossShop(
            df=sample_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
        )

        additional_kwargs = {
            "alpha": 0.7,
            "colors": ["red", "blue"],
            "custom_param": "value",
        }

        cross_shop.plot(**additional_kwargs)

        _, kwargs = mock_venn_plot.call_args
        for key, value in additional_kwargs.items():
            assert kwargs[key] == value
