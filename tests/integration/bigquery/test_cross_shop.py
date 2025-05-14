"""Integration tests for Cross Shop Analysis with BigQuery."""

from unittest.mock import MagicMock, patch

import pytest

from pyretailscience.analysis.cross_shop import CrossShop
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()

TWO_ELEMENT_TUPLE = 2
THREE_ELEMENT_TUPLE = 3
FLOAT_TOLERANCE = 1e-10


@pytest.fixture
def transactions_df(transactions_table):
    """Fetch transaction data for testing from BigQuery."""
    query = """
    SELECT
        customer_id,
        category_1_name,
        unit_spend
    FROM
        test_data.transactions
    WHERE
        category_1_name IN ('Jeans', 'Shoes', 'Dresses', 'Hats')
    LIMIT 100
    """
    return transactions_table.sql(query).execute()


def test_calc_cross_shop_two_groups_bigquery(transactions_df):
    """Test the _calc_cross_shop method with two groups using BigQuery data."""
    cross_shop_df = CrossShop._calc_cross_shop(
        transactions_df,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
    )

    assert "group_1" in cross_shop_df.columns
    assert "group_2" in cross_shop_df.columns
    assert "groups" in cross_shop_df.columns
    assert cols.unit_spend in cross_shop_df.columns

    assert all(isinstance(g, tuple) and len(g) == TWO_ELEMENT_TUPLE for g in cross_shop_df["groups"])
    assert all(cross_shop_df["group_1"].isin([0, 1]))
    assert all(cross_shop_df["group_2"].isin([0, 1]))


def test_calc_cross_shop_three_groups_bigquery(transactions_df):
    """Test the _calc_cross_shop method with three groups using BigQuery data."""
    cross_shop_df = CrossShop._calc_cross_shop(
        transactions_df,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
    )

    assert "group_1" in cross_shop_df.columns
    assert "group_2" in cross_shop_df.columns
    assert "group_3" in cross_shop_df.columns
    assert "groups" in cross_shop_df.columns
    assert cols.unit_spend in cross_shop_df.columns

    assert all(isinstance(g, tuple) and len(g) == THREE_ELEMENT_TUPLE for g in cross_shop_df["groups"])
    assert all(cross_shop_df["group_1"].isin([0, 1]))
    assert all(cross_shop_df["group_2"].isin([0, 1]))
    assert all(cross_shop_df["group_3"].isin([0, 1]))


def test_calc_cross_shop_three_groups_customer_id_nunique_bigquery(transactions_df):
    """Test the _calc_cross_shop method with three groups and customer_id as the value column using BigQuery data."""
    cross_shop_df = CrossShop._calc_cross_shop(
        transactions_df,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        group_3_col="category_1_name",
        group_3_val="Dresses",
        value_col=cols.customer_id,
        agg_func="nunique",
    )

    assert "group_1" in cross_shop_df.columns
    assert "group_2" in cross_shop_df.columns
    assert "group_3" in cross_shop_df.columns
    assert "groups" in cross_shop_df.columns
    assert cols.customer_id in cross_shop_df.columns

    assert cross_shop_df.index.nunique() == cross_shop_df.index.size


def test_calc_cross_shop_table_bigquery(transactions_df):
    """Test the _calc_cross_shop_table method with BigQuery data."""
    cross_shop_df = CrossShop._calc_cross_shop(
        transactions_df,
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

    assert "groups" in cross_shop_table.columns
    assert cols.unit_spend in cross_shop_table.columns
    assert "percent" in cross_shop_table.columns

    assert abs(cross_shop_table["percent"].sum() - 1.0) < FLOAT_TOLERANCE

    expected_combinations = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]

    actual_combinations = set(map(tuple, cross_shop_table["groups"]))
    assert all(combo in expected_combinations for combo in actual_combinations)


def test_calc_cross_shop_table_customer_id_nunique_bigquery(transactions_df):
    """Test the _calc_cross_shop_table method with customer_id as the value column using BigQuery data."""
    cross_shop_df = CrossShop._calc_cross_shop(
        transactions_df,
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

    assert "groups" in cross_shop_table.columns
    assert cols.customer_id in cross_shop_table.columns
    assert "percent" in cross_shop_table.columns

    assert abs(cross_shop_table["percent"].sum() - 1.0) < FLOAT_TOLERANCE

    assert all(isinstance(val, int | float) and val.is_integer() for val in cross_shop_table[cols.customer_id])


def test_calc_cross_shop_invalid_group_3_bigquery(transactions_df):
    """Test that _calc_cross_shop raises ValueError if only one of group_3_col or group_3_val is provided with BigQuery data."""
    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated, then the other must be as well"):
        CrossShop._calc_cross_shop(
            transactions_df,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
        )

    with pytest.raises(ValueError, match="If group_3_col or group_3_val is populated, then the other must be as well"):
        CrossShop._calc_cross_shop(
            transactions_df,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_val="T-Shirts",
        )


def test_missing_columns_raises_bigquery(transactions_df):
    """Test that ValueError is raised when required columns are missing in df with BigQuery data."""
    bad_data = transactions_df.drop(columns=[cols.unit_spend])
    with pytest.raises(ValueError, match="The following columns are required but missing:"):
        CrossShop(
            df=bad_data,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
        )


def test_labels_length_mismatch_bigquery(transactions_df):
    """Test that ValueError is raised when labels length does not match group count with BigQuery data."""
    with pytest.raises(ValueError, match="The number of labels must be equal to the number of group indexes given"):
        CrossShop(
            df=transactions_df,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
            group_3_col="category_1_name",
            group_3_val="Dresses",
            labels=["A", "B"],
        )


@patch("pyretailscience.plots.venn.plot")
def test_plot_passes_parameters_correctly_bigquery(mock_venn_plot, transactions_df):
    """Test that the plot method passes parameters correctly to venn.plot with BigQuery data."""
    cross_shop = CrossShop(
        df=transactions_df,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        labels=["Jeans", "Shoes"],
    )

    title = "Test Title - BigQuery Data"
    source_text = "Source: BigQuery Test Data"
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


def test_plot_with_default_labels_bigquery(transactions_df):
    """Test that plot generates default alphabetical labels when none are provided with BigQuery data."""
    with patch("pyretailscience.plots.venn.plot") as mock_venn_plot:
        cross_shop = CrossShop(
            df=transactions_df,
            group_1_col="category_1_name",
            group_1_val="Jeans",
            group_2_col="category_1_name",
            group_2_val="Shoes",
        )

        cross_shop.plot()

        _, kwargs = mock_venn_plot.call_args
        assert kwargs["labels"] == ["A", "B"]


def test_plot_with_default_labels_three_groups_bigquery(transactions_df):
    """Test that plot generates default alphabetical labels for three groups with BigQuery data."""
    with patch("pyretailscience.plots.venn.plot") as mock_venn_plot:
        cross_shop = CrossShop(
            df=transactions_df,
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


def test_cross_shop_with_custom_value_col_and_agg_func_bigquery(transactions_df):
    """Test CrossShop initialization with custom value_col and agg_func using BigQuery data."""
    transactions_df["custom_value"] = transactions_df[cols.unit_spend] * 2

    cross_shop = CrossShop(
        df=transactions_df,
        group_1_col="category_1_name",
        group_1_val="Jeans",
        group_2_col="category_1_name",
        group_2_val="Shoes",
        value_col="custom_value",
        agg_func="mean",
    )

    assert "custom_value" in cross_shop.cross_shop_df.columns

    assert cross_shop.cross_shop_df["custom_value"].dtype in [float, int]
    assert not cross_shop.cross_shop_df["custom_value"].isna().all()
