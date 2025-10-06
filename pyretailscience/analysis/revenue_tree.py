"""Revenue Tree Analysis Module.

This module implements a Revenue Tree analysis for retail businesses. The Revenue Tree
is a hierarchical breakdown of factors contributing to overall revenue, allowing for
detailed analysis of sales performance and identification of areas for improvement.

Key Components of the Revenue Tree:

1. Revenue: The top-level metric, calculated as Customers * Revenue per Customer.

2. Revenue per Customer: Average revenue generated per customer, calculated as:
   Orders per Customer * Average Order Value.

3. Orders per Customer: Average number of orders placed by each customer.

4. Average Order Value: Average monetary value of each order, calculated as:
   Items per Order * Price per Item.

5. Items per Order: Average number of items in each order.

6. Price per Item: Average price of each item sold.

This module can be used to create, update, and analyze Revenue Tree data structures
for retail businesses, helping to identify key drivers of revenue changes and
inform strategic decision-making.
"""

import graphviz
import ibis
import pandas as pd

from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.plots.tree_diagram import TreeDiagram
from pyretailscience.plugin import plugin_manager


@plugin_manager.extensible
def calc_tree_kpis(
    df: pd.DataFrame,
    p1_index: list[bool] | pd.Series,
    p2_index: list[bool] | pd.Series,
) -> pd.DataFrame:
    """Calculate various key performance indicators (KPIs) for tree analysis.

    Args:
        df (pd.DataFrame): Input DataFrame containing relevant data.
        p1_index (list[bool] | pd.Series): Boolean index for period 1.
        p2_index (list[bool] | pd.Series): Boolean index for period 2.

    Returns:
        pd.DataFrame: A DataFrame with calculated KPI values, including differences
        and percentage differences between periods.
    """
    cols = ColumnHelper()
    required_cols = [cols.agg_customer_id, cols.agg_transaction_id, cols.agg_unit_spend]

    if cols.agg_unit_qty in df.columns:
        required_cols.append(cols.agg_unit_qty)

    df = df[required_cols].copy()
    df_cols = df.columns

    if cols.agg_unit_qty in df_cols:
        df[cols.calc_units_per_trans] = df[cols.agg_unit_qty] / df[cols.agg_transaction_id]
        df[cols.calc_price_per_unit] = df[cols.agg_unit_spend] / df[cols.agg_unit_qty]

    df[cols.calc_spend_per_cust] = df[cols.agg_unit_spend] / df[cols.agg_customer_id]
    df[cols.calc_spend_per_trans] = df[cols.agg_unit_spend] / df[cols.agg_transaction_id]
    df[cols.calc_trans_per_cust] = df[cols.agg_transaction_id] / df[cols.agg_customer_id]

    p1_df = df[p1_index]
    p1_df.columns = [col + "_" + get_option("column.suffix.period_1") for col in p1_df.columns]
    p2_df = df[p2_index]
    p2_df.columns = [col + "_" + get_option("column.suffix.period_2") for col in p2_df.columns]

    # When df only contains two periods than the indexes should be dropped for proper concatenation
    period_count = 2
    if len(df.index) == period_count:
        p1_df = p1_df.reset_index(drop=True)
        p2_df = p2_df.reset_index(drop=True)

    # fillna with 0 to handle cases when one time period isn't present
    df = pd.concat([p1_df, p2_df], axis=1).fillna(0)

    for col in [
        cols.agg_customer_id,
        cols.agg_transaction_id,
        cols.agg_unit_spend,
        cols.calc_spend_per_trans,
        cols.calc_trans_per_cust,
        cols.calc_spend_per_cust,
    ]:
        # Difference calculations
        df[col + "_" + get_option("column.suffix.difference")] = (
            df[col + "_" + get_option("column.suffix.period_2")] - df[col + "_" + get_option("column.suffix.period_1")]
        )

        # Percentage change calculations
        df[col + "_" + get_option("column.suffix.percent_difference")] = (
            df[col + "_" + get_option("column.suffix.difference")]
            / df[col + "_" + get_option("column.suffix.period_1")]
        )

    # Calculate price elasticity
    if cols.agg_unit_qty in df_cols:
        df[cols.calc_price_elasticity] = (
            (df[cols.agg_unit_qty_p2] - df[cols.agg_unit_qty_p1])
            / ((df[cols.agg_unit_qty_p2] + df[cols.agg_unit_qty_p1]) / 2)
        ) / (
            (df[cols.calc_price_per_unit_p2] - df[cols.calc_price_per_unit_p1])
            / ((df[cols.calc_price_per_unit_p2] + df[cols.calc_price_per_unit_p1]) / 2)
        )

    # Calculate frequency elasticity
    df[cols.calc_frequency_elasticity] = (
        (df[cols.calc_trans_per_cust_p2] - df[cols.calc_trans_per_cust_p1])
        / ((df[cols.calc_trans_per_cust_p2] + df[cols.calc_trans_per_cust_p1]) / 2)
    ) / (
        (df[cols.calc_spend_per_cust_p2] - df[cols.calc_spend_per_cust_p1])
        / ((df[cols.calc_spend_per_cust_p2] + df[cols.calc_spend_per_cust_p1]) / 2)
    )

    # Contribution calculations
    df[cols.agg_customer_id_contrib] = (
        df[cols.agg_unit_spend_p2]
        - (df[cols.agg_customer_id_p1] * df[cols.calc_spend_per_cust_p2])
        - ((df[cols.agg_customer_id_diff] * df[cols.calc_spend_per_cust_diff]) / 2)
    )
    df[cols.calc_spend_per_cust_contrib] = (
        df[cols.agg_unit_spend_p2]
        - (df[cols.calc_spend_per_cust_p1] * df[cols.agg_customer_id_p2])
        - ((df[cols.agg_customer_id_diff] * df[cols.calc_spend_per_cust_diff]) / 2)
    )

    df[cols.calc_trans_per_cust_contrib] = (
        (
            df[cols.calc_spend_per_cust_p2]
            - (df[cols.calc_trans_per_cust_p1] * df[cols.calc_spend_per_trans_p2])
            - ((df[cols.calc_trans_per_cust_diff] * df[cols.calc_spend_per_trans_diff]) / 2)
        )
        * df[cols.agg_customer_id_p2]
    ) - ((df[cols.agg_customer_id_diff] * df[cols.calc_spend_per_cust_diff]) / 4)

    df[cols.calc_spend_per_trans_contrib] = (
        (
            df[cols.calc_spend_per_cust_p2]
            - (df[cols.calc_spend_per_trans_p1] * df[cols.calc_trans_per_cust_p2])
            - ((df[cols.calc_trans_per_cust_diff] * df[cols.calc_spend_per_trans_diff]) / 2)
        )
        * df[cols.agg_customer_id_p2]
    ) - ((df[cols.agg_customer_id_diff] * df[cols.calc_spend_per_cust_diff]) / 4)

    if cols.agg_unit_qty in df_cols:
        # Difference calculations
        for col in [
            cols.agg_unit_qty,
            cols.calc_units_per_trans,
            cols.calc_price_per_unit,
        ]:
            df[col + "_" + get_option("column.suffix.difference")] = (
                df[col + "_" + get_option("column.suffix.period_2")]
                - df[col + "_" + get_option("column.suffix.period_1")]
            )

        for col in [
            cols.agg_unit_qty,
            cols.calc_units_per_trans,
            cols.calc_price_per_unit,
        ]:
            df[col + "_" + get_option("column.suffix.percent_difference")] = (
                df[col + "_" + get_option("column.suffix.difference")]
                / df[col + "_" + get_option("column.suffix.period_1")]
            )

        df[cols.calc_price_per_unit_contrib] = (
            (
                (
                    df[cols.calc_spend_per_trans_p2]
                    - (df[cols.calc_price_per_unit_p1] * df[cols.calc_units_per_trans_p2])
                    - ((df[cols.calc_units_per_trans_diff] * df[cols.calc_price_per_unit_diff]) / 2)
                )
                * df[cols.calc_trans_per_cust_p2]
            )
            - ((df[cols.calc_trans_per_cust_diff] * df[cols.calc_spend_per_trans_diff]) / 4)
        ) * df[cols.agg_customer_id_p2] - ((df[cols.agg_customer_id_diff] * df[cols.calc_spend_per_cust_diff]) / 8)

        df[cols.calc_units_per_trans_contrib] = (
            (
                (
                    df[cols.calc_spend_per_trans_p2]
                    - (df[cols.calc_units_per_trans_p1] * df[cols.calc_price_per_unit_p2])
                    - ((df[cols.calc_units_per_trans_diff] * df[cols.calc_price_per_unit_diff]) / 2)
                )
                * df[cols.calc_trans_per_cust_p2]
            )
            - ((df[cols.calc_trans_per_cust_diff] * df[cols.calc_spend_per_trans_diff]) / 4)
        ) * df[cols.agg_customer_id_p2] - ((df[cols.agg_customer_id_diff] * df[cols.calc_spend_per_cust_diff]) / 8)

    cols = RevenueTree._get_final_col_order(include_quantity=cols.agg_unit_qty in df_cols)

    return df[cols]


@plugin_manager.extensible
class RevenueTree:
    """Revenue Tree Analysis Class."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str,
        p2_value: str,
        group_col: str | None = None,
    ) -> None:
        """Initialize the Revenue Tree Analysis Class.

        Args:
            df (pd.DataFrame | ibis.Table): The input DataFrame or ibis Table containing transaction data.
            period_col (str): The column representing the period.
            p1_value (str): The value representing the first period.
            p2_value (str): The value representing the second period.
            group_col (str, optional): The column to group the data by. Defaults to None.

        Raises:
            ValueError: If the required columns are not present in the DataFrame.
        """
        cols = ColumnHelper()

        required_cols = [
            cols.customer_id,
            cols.transaction_id,
            cols.unit_spend,
        ]
        if cols.unit_qty in df.columns:
            required_cols.append(cols.unit_qty)

        if group_col is not None:
            required_cols.append(group_col)

        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        df, p1_index, p2_index = self._agg_data(df, period_col, p1_value, p2_value, group_col)

        self.df = calc_tree_kpis(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
        )

    @staticmethod
    def _agg_data(
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str,
        p2_value: str,
        group_col: str | None = None,
    ) -> tuple[pd.DataFrame, list[bool], list[bool]]:
        cols = ColumnHelper()

        if isinstance(df, pd.DataFrame):
            df: ibis.Table = ibis.memtable(df)

        aggs = {
            cols.agg_customer_id: df[cols.customer_id].nunique(),
            cols.agg_transaction_id: df[cols.transaction_id].nunique(),
            cols.agg_unit_spend: df[cols.unit_spend].sum(),
        }
        if cols.unit_qty in df.columns:
            aggs[cols.agg_unit_qty] = df[cols.unit_qty].sum()

        group_by_cols = [group_col, period_col] if group_col else [period_col]
        df = pd.DataFrame(df.group_by(group_by_cols).aggregate(**aggs).execute())
        p1_df = df[df[period_col] == p1_value].drop(columns=[period_col])
        p2_df = df[df[period_col] == p2_value].drop(columns=[period_col])

        if group_col is not None:
            p1_df = p1_df.sort_values(by=group_col)
            p2_df = p2_df.sort_values(by=group_col)

        new_p1_index = [True] * len(p1_df) + [False] * len(p2_df)
        new_p2_index = [not i for i in new_p1_index]

        result_df = pd.concat([p1_df, p2_df], ignore_index=True)

        if group_col is None:
            result_df.index = ["p1", "p2"]
        else:
            result_df.set_index(group_col, inplace=True)
            result_df.index = pd.CategoricalIndex(result_df.index)
        return result_df, new_p1_index, new_p2_index

    @staticmethod
    def _get_final_col_order(include_quantity: bool) -> str:
        cols = ColumnHelper()
        col_order = [
            # Customers
            cols.agg_customer_id_p1,
            cols.agg_customer_id_p2,
            cols.agg_customer_id_diff,
            cols.agg_customer_id_pct_diff,
            cols.agg_customer_id_contrib,
            # Transactions
            cols.agg_transaction_id_p1,
            cols.agg_transaction_id_p2,
            cols.agg_transaction_id_diff,
            cols.agg_transaction_id_pct_diff,
            # Unit Spend
            cols.agg_unit_spend_p1,
            cols.agg_unit_spend_p2,
            cols.agg_unit_spend_diff,
            cols.agg_unit_spend_pct_diff,
            # Spend / Customer
            cols.calc_spend_per_cust_p1,
            cols.calc_spend_per_cust_p2,
            cols.calc_spend_per_cust_diff,
            cols.calc_spend_per_cust_pct_diff,
            cols.calc_spend_per_cust_contrib,
            # Transactions / Customer
            cols.calc_trans_per_cust_p1,
            cols.calc_trans_per_cust_p2,
            cols.calc_trans_per_cust_diff,
            cols.calc_trans_per_cust_pct_diff,
            cols.calc_trans_per_cust_contrib,
            # Spend / Transaction
            cols.calc_spend_per_trans_p1,
            cols.calc_spend_per_trans_p2,
            cols.calc_spend_per_trans_diff,
            cols.calc_spend_per_trans_pct_diff,
            cols.calc_spend_per_trans_contrib,
            # Elasticity
            cols.calc_frequency_elasticity,
        ]

        if include_quantity:
            col_order.extend(
                [
                    # Unit Quantity
                    cols.agg_unit_qty_p1,
                    cols.agg_unit_qty_p2,
                    cols.agg_unit_qty_diff,
                    cols.agg_unit_qty_pct_diff,
                    # Quantity / Transaction
                    cols.calc_units_per_trans_p1,
                    cols.calc_units_per_trans_p2,
                    cols.calc_units_per_trans_diff,
                    cols.calc_units_per_trans_pct_diff,
                    cols.calc_units_per_trans_contrib,
                    # Price / Unit
                    cols.calc_price_per_unit_p1,
                    cols.calc_price_per_unit_p2,
                    cols.calc_price_per_unit_diff,
                    cols.calc_price_per_unit_pct_diff,
                    cols.calc_price_per_unit_contrib,
                    # Price Elasticity
                    cols.calc_price_elasticity,
                ],
            )

        return col_order

    def draw_tree(
        self,
        value_labels: tuple[str, str] | None = None,
        unit_spend_label: str = "Revenue",
        customer_id_label: str = "Customers",
        spend_per_customer_label: str = "Spend / Customer",
        transactions_per_customer_label: str = "Visits / Customer",
        spend_per_transaction_label: str = "Spend / Visit",
        units_per_transaction_label: str = "Units / Visit",
        price_per_unit_label: str = "Price / Unit",
        human_format: bool = False,
    ) -> graphviz.Digraph:
        """Draw the Revenue Tree graph as a Graphviz visualization.

        Args:
            value_labels (tuple[str, str], optional): Labels for the value columns. Defaults to None. When None, the
                default labels of Current Period and Previous Period are used for P1 and P2.
            unit_spend_label (str, optional): Label for the Revenue column. Defaults to "Revenue".
            customer_id_label (str, optional): Label for the Customers column. Defaults to "Customers".
            spend_per_customer_label (str, optional): Label for the Spend / Customer column. Defaults to
                "Spend / Customer".
            transactions_per_customer_label (str, optional): Label for the Visits / Customer column. Defaults to
                "Visits / Customer".
            spend_per_transaction_label (str, optional): Label for the Spend / Visit column. Defaults to
                "Spend / Visit".
            units_per_transaction_label (str, optional): Label for the Units / Visit column. Defaults to
                "Units / Visit".
            price_per_unit_label (str, optional): Label for the Price / Unit column. Defaults to
                "Price / Unit".
            human_format (bool, optional): Whether to use human-readable formatting. Defaults to False.

        Returns:
            graphviz.Digraph: The Graphviz visualization of the Revenue Tree.
        """
        cols = ColumnHelper()
        graph_data = self.df.iloc[0].to_dict()
        tree = TreeDiagram()

        # Add Revenue node
        tree.add_node(
            name="agg_unit_spend",
            title=unit_spend_label,
            p2_value=graph_data[cols.agg_unit_spend_p2],
            p1_value=graph_data[cols.agg_unit_spend_p1],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Add Customers node
        tree.add_node(
            name="agg_customer_id",
            title=customer_id_label,
            p2_value=graph_data[cols.agg_customer_id_p2],
            p1_value=graph_data[cols.agg_customer_id_p1],
            contrib_value=graph_data[cols.agg_customer_id_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Add Spend / Customer node
        tree.add_node(
            name="calc_spend_per_customer",
            title=spend_per_customer_label,
            p2_value=graph_data[cols.calc_spend_per_cust_p2],
            p1_value=graph_data[cols.calc_spend_per_cust_p1],
            contrib_value=graph_data[cols.calc_spend_per_cust_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Add Visits / Customer node
        tree.add_node(
            name="calc_transactions_per_customer",
            title=transactions_per_customer_label,
            p2_value=graph_data[cols.calc_trans_per_cust_p2],
            p1_value=graph_data[cols.calc_trans_per_cust_p1],
            contrib_value=graph_data[cols.calc_trans_per_cust_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Add Spend / Visit node
        tree.add_node(
            name="calc_spend_per_transaction",
            title=spend_per_transaction_label,
            p2_value=graph_data[cols.calc_spend_per_trans_p2],
            p1_value=graph_data[cols.calc_spend_per_trans_p1],
            contrib_value=graph_data[cols.calc_spend_per_trans_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Add edges for the main tree structure
        tree.add_edge("agg_unit_spend", "calc_spend_per_customer")
        tree.add_edge("agg_unit_spend", "agg_customer_id")
        tree.add_edge("calc_spend_per_customer", "calc_transactions_per_customer")
        tree.add_edge("calc_spend_per_customer", "calc_spend_per_transaction")

        # Add quantity-related nodes if quantity data is available
        if cols.agg_unit_qty_p1 in graph_data:
            # Add Units / Visit node
            tree.add_node(
                name="calc_units_per_transaction",
                title=units_per_transaction_label,
                p2_value=graph_data[cols.calc_units_per_trans_p2],
                p1_value=graph_data[cols.calc_units_per_trans_p1],
                contrib_value=graph_data[cols.calc_units_per_trans_contrib],
                value_labels=value_labels,
                human_format=human_format,
            )

            # Add Price / Unit node
            tree.add_node(
                name="calc_price_per_unit",
                title=price_per_unit_label,
                p2_value=graph_data[cols.calc_price_per_unit_p2],
                p1_value=graph_data[cols.calc_price_per_unit_p1],
                contrib_value=graph_data[cols.calc_price_per_unit_contrib],
                value_labels=value_labels,
                human_format=human_format,
            )

            # Add edges for quantity-related nodes
            tree.add_edge("calc_spend_per_transaction", "calc_units_per_transaction")
            tree.add_edge("calc_spend_per_transaction", "calc_price_per_unit")

        return tree.render()
