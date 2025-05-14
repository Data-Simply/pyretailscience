"""Integration tests for Revenue Tree Analysis with BigQuery."""

import math

import pandas as pd
import pytest

from pyretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis
from pyretailscience.options import ColumnHelper

EXPECTED_LENGTH_2 = 2
EXPECTED_LENGTH_5 = 5
EXPECTED_SUM_3 = 3
EXPECTED_SUM_2 = 2


class TestRevenueTreeBigQuery:
    """Test the RevenueTree class with BigQuery integration."""

    @pytest.fixture
    def cols(self):
        """Return a ColumnHelper instance."""
        return ColumnHelper()

    @pytest.fixture
    def sample_transactions_df(self, transactions_table):
        """Get a sample of transactions from BigQuery."""
        query = transactions_table.limit(1000)
        return query.execute()

    def test_dataframe_missing_required_columns(self, cols: ColumnHelper, transactions_table):
        """Test that an error is raised when the DataFrame is missing required columns."""
        limited_cols = [
            "customer_id",
            "transaction_date",
        ]
        query = transactions_table.select(limited_cols).limit(10)
        df = query.execute()
        df["period"] = ["P1", "P2"] * (len(df) // 2)

        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2")
        assert "The following columns are required but missing:" in str(excinfo.value)

    def test_dataframe_missing_group_col(self, cols: ColumnHelper, transactions_table):
        """Test that an error is raised when the DataFrame is missing the group_col."""
        required_cols = [
            "customer_id",
            "unit_spend",
            "transaction_date",
            "transaction_id",
        ]
        query = transactions_table.select(required_cols).limit(10)
        df = query.execute()
        df["period"] = ["P1", "P2"] * (len(df) // 2)

        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col="brand_id")
        assert "The following columns are required but missing:" in str(excinfo.value)

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_no_group(self, cols: ColumnHelper, include_quantity: bool, transactions_table):
        """Test the _agg_data method with no group_col using BigQuery data."""
        columns = [
            "customer_id",
            "transaction_id",
            "unit_spend",
            "transaction_date",
        ]

        if include_quantity:
            columns.append("unit_quantity")

        query = transactions_table.select(columns).limit(6)
        df = query.execute()

        middle_index = len(df) // 2
        df["period"] = ["P1"] * middle_index + ["P2"] * (len(df) - middle_index)

        df["customer_id"] = df["customer_id"].astype(int)
        df["transaction_id"] = df["transaction_id"].astype(int)
        df["unit_spend"] = df["unit_spend"].astype(float)

        if include_quantity:
            df[cols.unit_qty] = df["unit_quantity"].astype(int)

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )

        assert len(result_df) == EXPECTED_LENGTH_2
        assert result_df.index.tolist() == ["p1", "p2"]
        assert all(
            col in result_df.columns
            for col in [
                cols.agg_customer_id,
                cols.agg_transaction_id,
                cols.agg_unit_spend,
            ]
        )

        if include_quantity:
            assert cols.agg_unit_qty in result_df.columns

        assert len(new_p1_index) == EXPECTED_LENGTH_2
        assert new_p1_index == [True, False]

        assert len(new_p2_index) == EXPECTED_LENGTH_2
        assert new_p2_index == [False, True]

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_with_group(self, cols: ColumnHelper, include_quantity: bool, transactions_table):
        """Test the _agg_data method with a group_col using BigQuery data."""
        columns = [
            "customer_id",
            "transaction_id",
            "unit_spend",
            "transaction_date",
            "brand_id",
        ]

        if include_quantity:
            columns.append("unit_quantity")

        query = transactions_table.select(columns).limit(20)
        df = query.execute()

        top_brands = df["brand_id"].value_counts().nlargest(2).index.tolist()
        df = df[df["brand_id"].isin(top_brands)].reset_index(drop=True)

        df = df.head(6)

        middle_index = len(df) // 2
        df["period"] = ["P1"] * middle_index + ["P2"] * (len(df) - middle_index)

        df["customer_id"] = df["customer_id"].astype(int)
        df["transaction_id"] = df["transaction_id"].astype(int)
        df["unit_spend"] = df["unit_spend"].astype(float)

        if include_quantity:
            df[cols.unit_qty] = df["unit_quantity"].astype(int)

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="brand_id",
        )

        assert result_df.index.name == "brand_id"
        assert isinstance(result_df.index, pd.CategoricalIndex)
        assert all(
            col in result_df.columns
            for col in [
                cols.agg_customer_id,
                cols.agg_transaction_id,
                cols.agg_unit_spend,
            ]
        )

        if include_quantity:
            assert cols.agg_unit_qty in result_df.columns

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_contrib_sum(self, cols: ColumnHelper, include_quantity: bool, transactions_table):
        """Test that the contributions add up to the total using BigQuery data."""
        columns = [
            "customer_id",
            "transaction_id",
            "unit_spend",
            "transaction_date",
        ]

        if include_quantity:
            columns.append("unit_quantity")

        query = transactions_table.select(columns).limit(6)
        df = query.execute()

        middle_index = len(df) // 2
        df["period"] = ["P1"] * middle_index + ["P2"] * (len(df) - middle_index)

        df["customer_id"] = df["customer_id"].astype(int)
        df["transaction_id"] = df["transaction_id"].astype(int)
        df["unit_spend"] = df["unit_spend"].astype(float)

        if include_quantity:
            df[cols.unit_qty] = df["unit_quantity"].astype(int)

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )

        tree_index = 0
        tree_data = rt.df.to_dict(orient="records")[tree_index]

        spend_per_trans_contrib = tree_data[cols.calc_spend_per_trans_contrib]
        spend_per_cust_contrib = tree_data[cols.calc_spend_per_cust_contrib]
        trans_per_cust_contrib = tree_data[cols.calc_trans_per_cust_contrib]
        cust_contrib = tree_data[cols.agg_customer_id_contrib]

        unit_spend_diff = tree_data[cols.agg_unit_spend_diff]

        assert math.isclose(trans_per_cust_contrib + spend_per_trans_contrib, spend_per_cust_contrib)
        assert math.isclose(cust_contrib + spend_per_cust_contrib, unit_spend_diff)

        assert (
            tree_data[cols.agg_unit_spend_p2] - tree_data[cols.agg_unit_spend_p1] == tree_data[cols.agg_unit_spend_diff]
        )
        assert (
            tree_data[cols.agg_customer_id_p2] - tree_data[cols.agg_customer_id_p1]
            == tree_data[cols.agg_customer_id_diff]
        )
        assert (
            tree_data[cols.agg_transaction_id_p2] - tree_data[cols.agg_transaction_id_p1]
            == tree_data[cols.agg_transaction_id_diff]
        )
        assert (
            tree_data[cols.calc_spend_per_cust_p2] - tree_data[cols.calc_spend_per_cust_p1]
            == tree_data[cols.calc_spend_per_cust_diff]
        )
        assert (
            tree_data[cols.calc_trans_per_cust_p2] - tree_data[cols.calc_trans_per_cust_p1]
            == tree_data[cols.calc_trans_per_cust_diff]
        )

        if include_quantity:
            units_per_trans_contrib = tree_data[cols.calc_units_per_trans_contrib]
            price_per_unit_contrib = tree_data[cols.calc_price_per_unit_contrib]

            assert math.isclose(units_per_trans_contrib + price_per_unit_contrib, spend_per_trans_contrib)

            assert (
                tree_data[cols.agg_unit_qty_p2] - tree_data[cols.agg_unit_qty_p1] == tree_data[cols.agg_unit_qty_diff]
            )
            assert (
                tree_data[cols.calc_units_per_trans_p2] - tree_data[cols.calc_units_per_trans_p1]
                == tree_data[cols.calc_units_per_trans_diff]
            )
            assert (
                tree_data[cols.calc_price_per_unit_p2] - tree_data[cols.calc_price_per_unit_p1]
                == tree_data[cols.calc_price_per_unit_diff]
            )
        else:
            assert cols.calc_units_per_trans_contrib not in tree_data
            assert cols.calc_price_per_unit_contrib not in tree_data

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_p1_only_group(self, cols: ColumnHelper, include_quantity: bool, transactions_table):
        """Test the _agg_data method with a group_col and some groups only in P1 period."""
        columns = [
            "customer_id",
            "transaction_id",
            "unit_spend",
            "brand_id",
        ]

        if include_quantity:
            columns.append("unit_quantity")

        query = transactions_table.select(columns).limit(20)
        df = query.execute()

        unique_brands = df["brand_id"].unique()[:3]

        test_df = pd.DataFrame(
            {
                "brand_id": [
                    unique_brands[0],
                    unique_brands[0],
                    unique_brands[0],
                    unique_brands[1],
                    unique_brands[1],
                    unique_brands[1],
                    unique_brands[2],
                ],
                "customer_id": [1, 2, 3, 4, 5, 6, 7],
                "transaction_id": [1, 2, 3, 4, 5, 6, 7],
                "unit_spend": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2", "P1"],
            },
        )

        if include_quantity:
            test_df[cols.unit_qty] = [1, 2, 3, 4, 5, 6, 7]

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df=test_df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="brand_id",
        )

        assert len(result_df) == EXPECTED_LENGTH_5
        assert result_df.index.name == "brand_id"

        assert unique_brands[2] in result_df.index
        assert len(new_p1_index) == EXPECTED_LENGTH_5
        assert len(new_p2_index) == EXPECTED_LENGTH_5

        assert sum(new_p1_index) == EXPECTED_SUM_3
        assert sum(new_p2_index) == EXPECTED_SUM_2

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_calc_tree_kpis_basic(self, cols: ColumnHelper, include_quantity: bool):
        """Test basic KPI calculations with data structured like BigQuery results."""
        df = pd.DataFrame(
            {
                cols.agg_customer_id: [3, 3],
                cols.agg_transaction_id: [3, 3],
                cols.agg_unit_spend: [900.0, 1100.0],
            },
            index=["p1", "p2"],
        )

        if include_quantity:
            df[cols.agg_unit_qty] = [10, 12]

        p1_index = [True, False]
        p2_index = [False, True]

        result = calc_tree_kpis(df, p1_index, p2_index)

        assert isinstance(result, pd.DataFrame)

        expected_columns = [
            "customers_diff",
            "customers_pct_diff",
            "transactions_diff",
            "transactions_pct_diff",
            "spend_diff",
            "spend_pct_diff",
            "spend_per_transaction_diff",
            "spend_per_transaction_pct_diff",
            "transactions_per_customer_diff",
            "transactions_per_customer_pct_diff",
            "spend_per_customer_diff",
            "spend_per_customer_pct_diff",
            "frequency_elasticity",
        ]
        for col in expected_columns:
            assert col in result.columns

        if include_quantity:
            q1, q2 = 10, 12
            p1, p2 = 900.0 / q1, 1100.0 / q2
            expected_elasticity = ((q2 - q1) / ((q2 + q1) / 2)) / ((p2 - p1) / ((p2 + p1) / 2))

            assert round(result["price_elasticity"].iloc[0], 6) == round(expected_elasticity, 6)
