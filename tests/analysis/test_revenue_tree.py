"""Tests for the RevenueTree class."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis
from pyretailscience.options import ColumnHelper, option_context


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

    def test_dataframe_missing_group_cols_list(self, cols: ColumnHelper):
        """Test that an error is raised when the DataFrame is missing group_col columns from a list."""
        data = {
            cols.customer_id: [1, 2, 3],
            cols.transaction_id: [1, 2, 3],
            cols.unit_spend: [100, 200, 300],
            cols.transaction_date: ["2023-01-01", "2023-06-02", "2023-01-03"],
            "period": ["P1", "P2", "P1"],
            "region": ["North", "South", "North"],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col=["region", "store"])
        assert "The following columns are required but missing:" in str(excinfo.value)
        assert "store" in str(excinfo.value)

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
            group_col=["group_id"],
        )

        # Verify single column produces CategoricalIndex, not MultiIndex
        assert isinstance(result_df.index, pd.CategoricalIndex)
        assert not isinstance(result_df.index, pd.MultiIndex)

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
            group_col=["group_id"],
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

    def test_with_custom_column_names(self, cols: ColumnHelper):
        """Test RevenueTree with custom column names."""
        custom_columns = {
            "column.transaction_id": "txn_id",
            "column.customer_id": "cust_id",
            "column.unit_spend": "spend_amt",
            "column.unit_quantity": "qty",
        }

        data = {
            "cust_id": [1, 2, 3, 4],
            "txn_id": [101, 102, 103, 104],
            "spend_amt": [100.0, 200.0, 150.0, 300.0],
            "qty": [2, 3, 1, 4],
            "period": ["P1", "P1", "P2", "P2"],
        }
        df = pd.DataFrame(data)

        with option_context(*[item for pair in custom_columns.items() for item in pair]):
            rt = RevenueTree(
                df=df,
                period_col="period",
                p1_value="P1",
                p2_value="P2",
            )

            assert not rt.df.empty, "Result should not be empty"
            expected_columns = [
                cols.agg_customer_id_p1,
                cols.agg_customer_id_p2,
                cols.agg_transaction_id_p1,
                cols.agg_transaction_id_p2,
                cols.agg_unit_spend_p1,
                cols.agg_unit_spend_p2,
                cols.agg_unit_qty_p1,
                cols.agg_unit_qty_p2,
            ]

            for col in expected_columns:
                assert col in rt.df.columns, f"Expected column {col} missing from output"

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_with_multi_group_cols(self, cols: ColumnHelper, include_quantity: bool):
        """Test the _agg_data method with multiple group columns."""
        df = pd.DataFrame(
            {
                "region": ["North", "North", "North", "South", "South", "South"],
                "store": ["A", "A", "A", "B", "B", "B"],
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

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6]

        result_df, new_p1_index, new_p2_index = RevenueTree._agg_data(
            df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col=["region", "store"],
        )

        # Verify MultiIndex structure
        assert isinstance(result_df.index, pd.MultiIndex)
        assert result_df.index.names == ["region", "store"]
        expected_num_index_levels = 2
        assert len(result_df.index.levels) == expected_num_index_levels

        # Verify data
        expected_num_rows = 4  # 2 groups * 2 periods
        assert len(result_df) == expected_num_rows
        assert new_p1_index == [True, True, False, False]
        assert new_p2_index == [False, False, True, True]

        # Verify aggregations are correct
        assert result_df[cols.agg_customer_id].tolist() == [2, 1, 1, 2]
        assert result_df[cols.agg_transaction_id].tolist() == [2, 1, 1, 2]
        assert result_df[cols.agg_unit_spend].tolist() == [400.0, 500.0, 200.0, 1000.0]

        if include_quantity:
            assert result_df[cols.agg_unit_qty].tolist() == [4, 5, 2, 10]

    def test_revenue_tree_with_multi_group_cols(self, cols: ColumnHelper):
        """Test RevenueTree end-to-end with multiple group columns."""
        df = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South"],
                "store": ["A", "A", "B", "B"],
                cols.customer_id: [1, 2, 3, 4],
                cols.transaction_id: [1, 2, 3, 4],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0],
                "period": ["P1", "P2", "P1", "P2"],
            },
        )

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col=["region", "store"],
        )

        # Verify MultiIndex
        assert isinstance(rt.df.index, pd.MultiIndex)
        assert rt.df.index.names == ["region", "store"]
        expected_num_combinations = 2
        assert len(rt.df) == expected_num_combinations  # 2 unique combinations

        # Verify we have all the expected columns
        assert cols.agg_customer_id_p1 in rt.df.columns
        assert cols.agg_customer_id_p2 in rt.df.columns
        assert cols.agg_unit_spend_p1 in rt.df.columns
        assert cols.agg_unit_spend_p2 in rt.df.columns

    @pytest.mark.parametrize("include_qty", [True, False])
    def test_draw_tree_matplotlib(self, cols: ColumnHelper, include_qty: bool):
        """Test that draw_tree() returns a matplotlib Axes and renders correctly."""
        data = {
            cols.customer_id: [1, 2, 3, 1, 2, 3],
            cols.transaction_id: [1, 2, 3, 4, 5, 6],
            cols.unit_spend: [100.0, 150.0, 200.0, 120.0, 180.0, 240.0],
            "period": ["P1", "P1", "P1", "P2", "P2", "P2"],
        }

        if include_qty:
            data[cols.unit_qty] = [10, 15, 20, 12, 18, 24]

        df = pd.DataFrame(data)

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )

        # Test with default parameters
        ax = rt.draw_tree()
        assert isinstance(ax, Axes)
        assert ax.get_title() == ""  # TreeGrid doesn't set a title by default

        # Clean up
        plt.close()

    @pytest.mark.parametrize("include_qty", [True, False])
    def test_draw_tree_auto_layout(self, cols: ColumnHelper, include_qty: bool):
        """draw_tree should render with auto-layout without errors."""
        data = {
            cols.customer_id: [1, 2, 3, 1, 2, 3],
            cols.transaction_id: [1, 2, 3, 4, 5, 6],
            cols.unit_spend: [100.0, 150.0, 200.0, 120.0, 180.0, 240.0],
            "period": ["P1", "P1", "P1", "P2", "P2", "P2"],
        }

        if include_qty:
            data[cols.unit_qty] = [10, 15, 20, 12, 18, 24]

        df = pd.DataFrame(data)

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )

        ax = rt.draw_tree()
        assert isinstance(ax, Axes)
        plt.close()

        # Test with custom node dimensions
        ax = rt.draw_tree(node_width=4.0, node_height=3.0)
        assert isinstance(ax, Axes)
        plt.close()

        # Test with custom value labels
        ax = rt.draw_tree(value_labels=("Current", "Previous"))
        assert isinstance(ax, Axes)
        plt.close()

        # Test with custom node labels
        ax = rt.draw_tree(
            unit_spend_label="Sales",
            customer_id_label="Shoppers",
            spend_per_customer_label="Sales / Shopper",
            transactions_per_customer_label="Trips / Shopper",
            spend_per_transaction_label="Sales / Trip",
            units_per_transaction_label="Items / Trip",
            price_per_unit_label="Price / Item",
        )
        assert isinstance(ax, Axes)
        plt.close()

    def test_draw_tree_with_row_index(self, cols: ColumnHelper):
        """Test that draw_tree() can visualize a specific row from a multi-group RevenueTree."""
        data = {
            "region": ["North", "North", "North", "South", "South", "South"],
            cols.customer_id: [1, 2, 1, 3, 4, 3],
            cols.transaction_id: [1, 2, 3, 4, 5, 6],
            cols.unit_spend: [100.0, 150.0, 120.0, 200.0, 250.0, 220.0],
            cols.unit_qty: [10, 15, 12, 20, 25, 22],
            "period": ["P1", "P1", "P2", "P1", "P1", "P2"],
        }

        df = pd.DataFrame(data)

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="region",
        )

        # Should have 2 rows (North and South)
        expected_regions = ["North", "South"]
        assert len(rt.df) == len(expected_regions)
        assert rt.df.index.tolist() == expected_regions

        # Draw first row (North region)
        ax = rt.draw_tree(row_index=0)
        assert isinstance(ax, Axes)
        # Check that the North region's revenue value appears in the plot
        text_strings = [t.get_text() for t in ax.texts]
        # North P2 revenue is 120, P1 revenue is 250 (100 + 150)
        # The formatted values should appear in the text
        assert any("120" in s for s in text_strings), "North region P2 revenue should appear"
        plt.close()

        # Draw second row (South region)
        ax = rt.draw_tree(row_index=1)
        assert isinstance(ax, Axes)
        text_strings = [t.get_text() for t in ax.texts]
        # South P2 revenue is 220, P1 revenue is 450 (200 + 250)
        assert any("220" in s for s in text_strings), "South region P2 revenue should appear"
        plt.close()

        # Test that out of bounds raises IndexError
        with pytest.raises(IndexError):
            rt.draw_tree(row_index=2)
