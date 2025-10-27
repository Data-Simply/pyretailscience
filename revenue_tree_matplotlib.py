"""Module for creating revenue tree diagrams using matplotlib."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pyretailscience.plots.tree_diagram import DetailedTreeNode, SimpleTreeNode, TreeGrid


def create_revenue_tree() -> Axes:
    """Create the revenue tree diagram.

    Returns:
        The matplotlib axes object.

    """
    # Define tree structure with grid coordinates (col, row)
    # Grid positions use (col, row) with (0,0) at top-left
    tree_structure = {
        "total_sales": {
            "header": "Total Sales (TISP)",
            "percent": -31.5,
            "value1": "£889.5k",
            "value2": "£1.3m",
            "position": (1, 0),
            "children": ["non_card_sales", "ad_card_sales"],
        },
        "non_card_sales": {
            "header": "Non-Card Sales (TISP)",
            "percent": -37.4,
            "value1": "£241.7k",
            "value2": "£385.8k",
            "position": (0, 1),
            "children": [],
        },
        "ad_card_sales": {
            "header": "Ad Card Sales (TISP)",
            "percent": 15.0,
            "value1": "£647.8k",
            "value2": "£912.7k",
            "position": (2, 1),
            "children": ["num_customers", "av_customer_value"],
        },
        "num_customers": {
            "header": "Number of Customers",
            "percent": -14.9,
            "value1": "65.4k",
            "value2": "76.8k",
            "position": (1, 2),
            "children": [],
        },
        "av_customer_value": {
            "header": "Av. Customer Value",
            "percent": -16.6,
            "value1": "£9.91",
            "value2": "£11.88",
            "position": (3, 2),
            "children": ["av_customer_freq", "av_transaction_value"],
        },
        "av_customer_freq": {
            "header": "Av. Customer Frequency",
            "percent": -8.8,
            "value1": "1.17",
            "value2": "1.29",
            "position": (2, 3),
            "children": [],
        },
        "av_transaction_value": {
            "header": "Av. Transaction Value",
            "percent": -8.5,
            "value1": "£8.44",
            "value2": "£9.22",
            "position": (4, 3),
            "children": ["items_per_transaction", "av_item_value"],
        },
        "items_per_transaction": {
            "header": "Items Per Transaction",
            "percent": -0.9,
            "value1": "1.55",
            "value2": "1.56",
            "position": (3, 4),
            "children": [],
        },
        "av_item_value": {
            "header": "Av. Item Value",
            "percent": -7.6,
            "value1": "£5.45",
            "value2": "£5.90",
            "position": (5, 4),
            "children": [],
        },
    }

    # Create and render the tree grid
    tree_grid = TreeGrid(
        tree_structure=tree_structure,
        num_rows=5,
        num_cols=6,
        node_class=SimpleTreeNode,
    )

    ax = tree_grid.render()

    plt.tight_layout()
    plt.savefig("tree_diagram_matplotlib.png", dpi=300)

    return ax


def create_detailed_revenue_tree() -> Axes:
    """Create the revenue tree diagram with detailed nodes.

    Returns:
        The matplotlib axes object.

    """
    # Define tree structure with detailed information
    # Grid positions use (col, row) with (0,0) at top-left
    tree_structure = {
        "revenue": {
            "header": "Revenue",
            "percent": 21.5,
            "current_period": "15,705.00",
            "previous_period": "12,922.92",
            "diff": "2,782.08",
            # Contribution omitted for root node (would be same as diff)
            "position": (1, 0),
            "children": ["customers", "spend_per_customer"],
        },
        "customers": {
            "header": "Customers",
            "percent": 8.5,
            "current_period": "4.12",
            "previous_period": "3.80",
            "diff": "0.32",
            "contribution": "4.23M",
            "position": (0, 1),
            "children": [],
        },
        "spend_per_customer": {
            "header": "Spend / Customer",
            "percent": 12.1,
            "current_period": "3,810.92",
            "previous_period": "3,400.76",
            "diff": "410.16",
            "contribution": "4.45M",
            "position": (2, 1),
            "children": ["visits_per_customer", "spend_per_visit"],
        },
        "visits_per_customer": {
            "header": "Visits / Customer",
            "percent": -5.2,
            "current_period": "12.4",
            "previous_period": "13.1",
            "diff": "-0.7",
            "contribution": "-1.82M",
            "position": (1, 2),
            "children": [],
        },
        "spend_per_visit": {
            "header": "Spend / Visit",
            "percent": 18.3,
            "current_period": "307.33",
            "previous_period": "259.68",
            "diff": "47.65",
            "contribution": "6.27M",
            "position": (3, 2),
            "children": ["units_per_visit", "price_per_unit"],
        },
        "units_per_visit": {
            "header": "Units / Visit",
            "percent": 15.8,
            "current_period": "285.12",
            "previous_period": "246.22",
            "diff": "38.90",
            "contribution": "5.12M",
            "position": (2, 3),
            "children": [],
        },
        "price_per_unit": {
            "header": "Price / Unit",
            "percent": 0.5,
            "current_period": "22.21",
            "previous_period": "21.73",
            "diff": "0.48",
            "contribution": "1.15M",
            "position": (4, 3),
            "children": [],
        },
    }

    # Create and render the tree grid with DetailedTreeNode
    tree_grid = TreeGrid(
        tree_structure=tree_structure,
        num_rows=4,
        num_cols=5,
        node_class=DetailedTreeNode,
    )

    ax = tree_grid.render()

    plt.tight_layout()
    plt.savefig("tree_diagram_detailed.png", dpi=300)

    return ax


if __name__ == "__main__":
    # Create both trees
    create_revenue_tree()
    create_detailed_revenue_tree()
