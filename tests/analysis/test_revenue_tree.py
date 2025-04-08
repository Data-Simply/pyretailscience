"""Tests for the RevenueTree class."""

import math

import pandas as pd
import pytest

from pyretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis
from pyretailscience.options import ColumnHelper


class TestRevenueTree:
    """Test the RevenueTree class."""

    @pytest.fixture
    def cols(self):
        """Return a ColumnHelper instance."""
        return ColumnHelper()

    def test_dataframe_missing_required_columns(self, cols: ColumnHelper):
        """Test that an error is raised when the DataFrame is missing required columns."""
        data = {
            cols.customer_id: [1, 2, 3],
            cols.transaction_date: ["2023-01-01", "2023-06-02", "2023-01-03"],
            "period": ["P1", "P2", "P1"],
        }
        df = pd.DataFrame(data)
        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2")
        assert "The following columns are required but missing:" in str(excinfo.value)

    def test_dataframe_missing_group_col(self, cols: ColumnHelper):
        """Test that an error is raised when the DataFrame is missing the group_col."""
        data = {
            cols.customer_id: [1, 2, 3],
            cols.unit_spend: [100, 200, 300],
            cols.transaction_date: ["2023-01-01", "2023-06-02", "2023-01-03"],
            "period": ["P1", "P2", "P1"],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col="group_id")
        assert "The following columns are required but missing:" in str(excinfo.value)

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_no_group(self, cols: ColumnHelper, include_quantity: bool):
        """Test the _agg_data method with no group_col."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.transaction_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_date: [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-06",
                    "2023-01-02",
                    "2023-01-04",
                ],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2"],
            },
        )

        expected_df = pd.DataFrame(
            {
                cols.agg_customer_id: [3, 3],
                cols.agg_transaction_id: [3, 3],
                cols.agg_unit_spend: [900.0, 1200.0],
            },
            index=["p1", "p2"],
        )

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6]
            expected_df[cols.agg_unit_qty] = [9, 12]

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )

        pd.testing.assert_frame_equal(result_df, expected_df)
        assert new_p1_index == [True, False]
        assert new_p2_index == [False, True]

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_with_group(self, cols: ColumnHelper, include_quantity: bool):
        """Test the _agg_data method with a group_col."""
        df = pd.DataFrame(
            {
                "group_id": [1, 1, 1, 2, 2, 2],
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.transaction_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_date: [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-06",
                    "2023-01-02",
                    "2023-01-04",
                ],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2"],
            },
        )

        expected_df = pd.DataFrame(
            {
                "group_id": [1, 2, 1, 2],
                cols.agg_customer_id: [2, 1, 1, 2],
                cols.agg_transaction_id: [2, 1, 1, 2],
                cols.agg_unit_spend: [400.0, 500.0, 200.0, 1000.0],
            },
        ).set_index("group_id")
        expected_df.index = pd.CategoricalIndex(expected_df.index)

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6]
            expected_df[cols.agg_unit_qty] = [4, 5, 2, 10]

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="group_id",
        )

        pd.testing.assert_frame_equal(result_df, expected_df)
        assert new_p1_index == [True, True, False, False]
        assert new_p2_index == [False, False, True, True]

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_contrib_sum(self, cols: ColumnHelper, include_quantity: bool):
        """Test that the contributions add up to the total."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.transaction_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [150.0, 75.5, 220.0, 310.0, 450.0, 120.0],
                cols.transaction_date: [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-06",
                    "2023-01-02",
                    "2023-01-04",
                ],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2"],
            },
        )

        if include_quantity:
            df[cols.unit_qty] = [2, 1, 3, 4, 5, 2]

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
        assert (
            tree_data[cols.calc_spend_per_cust_p2] - tree_data[cols.calc_spend_per_cust_p1]
            == tree_data[cols.calc_spend_per_cust_diff]
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
    def test_agg_data_p1_only_group(self, cols: ColumnHelper, include_quantity: bool):
        """Test the _agg_data method with a group_col."""
        df = pd.DataFrame(
            {
                "group_id": [1, 1, 1, 2, 2, 2, 3],
                cols.customer_id: [1, 2, 3, 4, 5, 6, 7],
                cols.transaction_id: [1, 2, 3, 4, 5, 6, 7],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2", "P1"],
            },
        )

        expected_df = pd.DataFrame(
            {
                "group_id": [1, 2, 3, 1, 2],
                cols.agg_customer_id: [2, 1, 1, 1, 2],
                cols.agg_transaction_id: [2, 1, 1, 1, 2],
                cols.agg_unit_spend: [400.0, 500.0, 700.0, 200.0, 1000.0],
            },
        ).set_index("group_id")
        expected_df.index = pd.CategoricalIndex(expected_df.index)

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6, 7]
            expected_df[cols.agg_unit_qty] = [4, 5, 7, 2, 10]

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="group_id",
        )

        pd.testing.assert_frame_equal(result_df, expected_df)
        assert new_p1_index == [True, True, True, False, False]
        assert new_p2_index == [False, False, False, True, True]

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_contrib_sum_p1_only_group(self, cols: ColumnHelper, include_quantity: bool):
        """Test that the contributions add up to the total."""
        df = pd.DataFrame(
            {
                "group_id": [1, 1, 1, 2, 2, 2, 3],
                cols.customer_id: [1, 2, 3, 4, 5, 6, 7],
                cols.transaction_id: [1, 2, 3, 4, 5, 6, 7],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                cols.transaction_date: [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-06",
                    "2023-01-02",
                    "2023-01-04",
                    "2022-12-10",
                ],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2", "P1"],
            },
        )

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6, 7]

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
        assert (
            tree_data[cols.calc_spend_per_cust_p2] - tree_data[cols.calc_spend_per_cust_p1]
            == tree_data[cols.calc_spend_per_cust_diff]
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
    def test_calc_tree_kpis_basic(self, cols: ColumnHelper, include_quantity: bool):
        """Test basic KPI calculations."""
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
