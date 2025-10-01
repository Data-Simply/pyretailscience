"""Tree Diagram Module.

This module implements a TreeDiagram class for creating hierarchical tree visualizations
using Graphviz. The TreeDiagram class can be used by various analysis classes to create
reusable tree-based visualizations.
"""

import platform
import subprocess
from textwrap import dedent

import graphviz

from pyretailscience.options import ColumnHelper
from pyretailscience.plots.styles import graph_utils as gu
from pyretailscience.plots.styles.tailwind import COLORS


class TreeDiagram:
    """Class for creating hierarchical tree visualizations using Graphviz."""

    def __init__(self) -> None:
        """Initialize the TreeDiagram class."""
        self.graph = graphviz.Digraph()
        self.graph.attr("graph", bgcolor="transparent")

    @staticmethod
    def _check_graphviz_installation() -> bool:
        """Check if Graphviz is installed on the system.

        Returns:
            bool: True if Graphviz is installed, False otherwise.
        """
        system = platform.system().lower()
        try:
            subprocess.run(["dot", "-V"], check=True, stderr=subprocess.DEVNULL, shell=(system == "windows"))  # noqa: S603 S607
        except FileNotFoundError:
            return False
        except subprocess.CalledProcessError:
            return False

        return True

    def add_node(
        self,
        name: str,
        title: str,
        p2_value: float,
        p1_value: float,
        contrib_value: float | None = None,
        value_decimal_places: int = 2,
        diff_decimal_places: int = 2,
        pct_decimal_places: int = 1,
        value_labels: tuple[str, str] | None = None,
        show_diff: bool = True,
        value_suffix: str = "",
        human_format: bool = False,
    ) -> None:
        """Add a node to the tree.

        Args:
            name (str): The unique name/identifier for the node.
            title (str): The title to display on the node.
            p2_value (float): The current period value.
            p1_value (float): The previous period value.
            contrib_value (float | None, optional): The contribution value. Defaults to None.
            value_decimal_places (int, optional): Number of decimal places for values. Defaults to 2.
            diff_decimal_places (int, optional): Number of decimal places for differences. Defaults to 2.
            pct_decimal_places (int, optional): Number of decimal places for percentages. Defaults to 1.
            value_labels (tuple[str, str] | None, optional): Labels for the value columns. Defaults to None.
            show_diff (bool, optional): Whether to show the difference row. Defaults to True.
            value_suffix (str, optional): Suffix to append to values. Defaults to "".
            human_format (bool, optional): Whether to use human-readable formatting. Defaults to False.
        """
        if value_labels is None:
            value_labels = ("Current Period", "Previous Period")

        diff = p2_value - p1_value

        if human_format:
            p2_value_str = (gu.human_format(p2_value, 0, decimals=value_decimal_places) + " " + value_suffix).strip()
            p1_value_str = (gu.human_format(p1_value, 0, decimals=value_decimal_places) + " " + value_suffix).strip()
            diff_str = (gu.human_format(diff, 0, decimals=diff_decimal_places) + " " + value_suffix).strip()
        else:
            style = "," if isinstance(p2_value, int) else f",.{value_decimal_places}f"
            p2_value_str = f"{p2_value:{style}} {value_suffix}".strip()
            p1_value_str = f"{p1_value:{style}} {value_suffix}".strip()
            diff_str = f"{diff:{style}} {value_suffix}".strip()

        pct_diff_str = "N/A - Divide By 0" if p1_value == 0 else f"{diff / p1_value * 100:,.{pct_decimal_places}f}%"

        diff_color = "darkgreen" if diff >= 0 else "red"

        height = 1.5
        diff_html = ""
        if show_diff:
            diff_html = dedent(
                f"""
            <tr>
                <td align="right"><font color="white" face="arial"><b>Diff&nbsp;</b></font></td>
                <td bgcolor="white"><font color="{diff_color}" face="arial">{diff_str}</font></td>
            </tr>
            """,
            )
            height += 0.25

        contrib_html = ""
        if contrib_value is not None:
            contrib_str = gu.human_format(contrib_value, 0, decimals=value_decimal_places)
            contrib_color = "darkgreen" if diff >= 0 else "red"
            contrib_html = dedent(
                f"""
            <tr>
                <td align="right"><font color="white" face="arial"><b>Contribution&nbsp;</b></font></td>
                <td bgcolor="white"><font color="{contrib_color}" face="arial">{contrib_str}</font></td>
            </tr>
            """,
            )
            height += 0.25

        self.graph.node(
            name=name,
            shape="box",
            style="filled, rounded",
            color=COLORS["green"][500],
            width="4",
            height=str(height),
            align="center",
            label=dedent(
                f"""<
                <table border="0" align="center" width="100%">
                    <tr><td colspan="2"><font point-size="18" color="white" face="arial"><b>{title}</b></font></td></tr>
                    <tr>
                        <td width="150%"><font color="white" face="arial"><b>{value_labels[0]}</b></font></td>
                        <td width="150%"><font color="white" face="arial"><b>{value_labels[1]}</b></font></td>
                    </tr>
                    <tr>
                        <td bgcolor="white"><font face="arial">{p2_value_str}</font></td>
                        <td bgcolor="white"><font face="arial">{p1_value_str}</font></td>
                    </tr>
                    {diff_html}
                    <tr>
                        <td align="right"><font color="white" face="arial"><b>Pct Diff&nbsp;</b></font></td>
                        <td bgcolor="white"><font color="{diff_color}" face="arial">{pct_diff_str}</font></td>
                    </tr>
                    {contrib_html}
                </table>
                >""",
            ),
        )

    def add_edge(self, from_node: str, to_node: str, **edge_attrs: str) -> None:
        """Add an edge between two nodes.

        Args:
            from_node (str): The name of the source node.
            to_node (str): The name of the target node.
            **edge_attrs: Additional edge attributes to pass to Graphviz.
        """
        self.graph.edge(from_node, to_node, **edge_attrs)

    def render(self) -> graphviz.Digraph:
        """Return the completed graph.

        Returns:
            graphviz.Digraph: The completed Graphviz tree diagram.

        Raises:
            ImportError: If Graphviz is not installed on the system.
        """
        if not self._check_graphviz_installation():
            raise ImportError(
                "Graphviz is required to draw the tree graph. See here for installation instructions: "
                "https://github.com/xflr6/graphviz?tab=readme-ov-file#installation",
            )
        return self.graph

    def draw_tree(
        self,
        graph_data: dict,
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
            graph_data (dict): Dictionary containing the tree data.
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

        tree = TreeDiagram()

        tree.add_node(
            name="agg_unit_spend",
            title=unit_spend_label,
            p2_value=graph_data[cols.agg_unit_spend_p2],
            p1_value=graph_data[cols.agg_unit_spend_p1],
            value_labels=value_labels,
            human_format=human_format,
        )

        tree.add_node(
            name="agg_customer_id",
            title=customer_id_label,
            p2_value=graph_data[cols.agg_customer_id_p2],
            p1_value=graph_data[cols.agg_customer_id_p1],
            contrib_value=graph_data[cols.agg_customer_id_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Spend / Cust
        tree.add_node(
            name="calc_spend_per_customer",
            title=spend_per_customer_label,
            p2_value=graph_data[cols.calc_spend_per_cust_p2],
            p1_value=graph_data[cols.calc_spend_per_cust_p1],
            contrib_value=graph_data[cols.calc_spend_per_cust_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        # Visits / Customer
        tree.add_node(
            name="calc_transactions_per_customer",
            title=transactions_per_customer_label,
            p2_value=graph_data[cols.calc_trans_per_cust_p2],
            p1_value=graph_data[cols.calc_trans_per_cust_p1],
            contrib_value=graph_data[cols.calc_trans_per_cust_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )
        # Spend / Visit
        tree.add_node(
            name="calc_spend_per_transaction",
            title=spend_per_transaction_label,
            p2_value=graph_data[cols.calc_spend_per_trans_p2],
            p1_value=graph_data[cols.calc_spend_per_trans_p1],
            contrib_value=graph_data[cols.calc_spend_per_trans_contrib],
            value_labels=value_labels,
            human_format=human_format,
        )

        tree.add_edge("agg_unit_spend", "calc_spend_per_customer")
        tree.add_edge("agg_unit_spend", "agg_customer_id")

        tree.add_edge("calc_spend_per_customer", "calc_transactions_per_customer")
        tree.add_edge("calc_spend_per_customer", "calc_spend_per_transaction")

        if cols.agg_unit_qty_p1 in graph_data:
            # Units / Visit
            tree.add_node(
                name="calc_units_per_transaction",
                title=units_per_transaction_label,
                p2_value=graph_data[cols.calc_units_per_trans_p2],
                p1_value=graph_data[cols.calc_units_per_trans_p1],
                contrib_value=graph_data[cols.calc_units_per_trans_contrib],
                value_labels=value_labels,
                human_format=human_format,
            )

            # Price / Unit
            tree.add_node(
                name="calc_price_per_unit",
                title=price_per_unit_label,
                p2_value=graph_data[cols.calc_price_per_unit_p2],
                p1_value=graph_data[cols.calc_price_per_unit_p1],
                contrib_value=graph_data[cols.calc_price_per_unit_contrib],
                value_labels=value_labels,
                human_format=human_format,
            )

            tree.add_edge("calc_spend_per_transaction", "calc_units_per_transaction")
            tree.add_edge("calc_spend_per_transaction", "calc_price_per_unit")

        return tree.render()
