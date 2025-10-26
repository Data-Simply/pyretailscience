"""Module for creating revenue tree diagrams using matplotlib."""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path

from pyretailscience.plots.styles.styling_context import get_styling_context
from pyretailscience.plots.styles.tailwind import COLORS
from pyretailscience.plots.tree_diagram import BaseRoundedBox


class TreeNode(ABC):
    """Abstract base class for tree nodes."""

    # Subclasses must define these class attributes
    NODE_WIDTH: float
    NODE_HEIGHT: float

    def __init__(
        self,
        x: float,
        y: float,
        data: dict[str, Any],
    ) -> None:
        """Initialize the tree node.

        Args:
            x: X-coordinate of bottom-left corner.
            y: Y-coordinate of bottom-left corner.
            data: Dictionary containing node data. Each subclass defines required keys.

        """
        self.x = x
        self.y = y
        self._data = data

    @abstractmethod
    def render(self, ax: Axes) -> None:
        """Render the node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        ...


class SimpleTreeNode(TreeNode):
    """Simple tree node implementation with header and data sections.

    Required data keys:
        header: str - The header text
        percent: float - Percentage change value
        value1: str - First value text
        value2: str - Second value text
    """

    NODE_WIDTH = 3.0
    NODE_HEIGHT = 1.2

    @staticmethod
    def _get_color(percent_change: float) -> str:
        """Return color based on percent change thresholds.

        Green if > 1%, Red if < -1%, Grey if between -1% and 1%.

        Args:
            percent_change: Percentage change value.

        Returns:
            Hex color code as string.

        """
        if percent_change > 1:
            return COLORS["green"][500]
        if percent_change < -1:
            return COLORS["red"][500]
        return COLORS["gray"][500]

    def render(self, ax: Axes) -> None:
        """Render the node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        # Extract data from the data dict
        header = self._data["header"]
        percent = self._data["percent"]
        value1 = self._data["value1"]
        value2 = self._data["value2"]

        # Styling constants
        corner_radius = 0.15
        header_height_ratio = 0.4
        header_color = COLORS["blue"][800]  # "#1E3A8A"
        text_color = "white"
        value_vertical_offset = 0.1

        # Positioning fractions
        header_text_x_fraction = 1 / 2
        percent_text_x_fraction = 4 / 16
        value_text_x_fraction = 11 / 16

        styling_context = get_styling_context()

        # Get standard font properties
        semi_bold_font = styling_context.get_font_properties(styling_context.fonts.title_font)
        regular_font = styling_context.get_font_properties(styling_context.fonts.label_font)

        # Determine color based on percent change
        color = self._get_color(percent)

        # Header section
        header_height = self.NODE_HEIGHT * header_height_ratio
        header_box = BaseRoundedBox(
            (self.x, self.y + self.NODE_HEIGHT - header_height),
            self.NODE_WIDTH,
            header_height,
            top_radius=corner_radius,
            bottom_radius=0,
            facecolor=header_color,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(header_box)

        # Data section
        data_height = self.NODE_HEIGHT - header_height
        data_box = BaseRoundedBox(
            (self.x, self.y),
            self.NODE_WIDTH,
            data_height,
            top_radius=0,
            bottom_radius=corner_radius,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(data_box)

        ax.text(
            self.x + self.NODE_WIDTH * header_text_x_fraction,
            self.y + self.NODE_HEIGHT - header_height / 2,
            header,
            ha="center",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=styling_context.fonts.label_size,
            color=text_color,
        )

        ax.text(
            self.x + self.NODE_WIDTH * percent_text_x_fraction,
            self.y + data_height / 2,
            f"{percent:+.1f}%",
            ha="center",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=styling_context.fonts.title_size,
            color=text_color,
        )

        ax.text(
            self.x + self.NODE_WIDTH * value_text_x_fraction,
            self.y + data_height / 2 + value_vertical_offset,
            value1,
            ha="left",
            va="center",
            fontproperties=regular_font,
            fontsize=styling_context.fonts.label_size,
            color=text_color,
        )
        ax.text(
            self.x + self.NODE_WIDTH * value_text_x_fraction,
            self.y + data_height / 2 - value_vertical_offset,
            value2,
            ha="left",
            va="center",
            fontproperties=regular_font,
            fontsize=styling_context.fonts.label_size,
            color=text_color,
        )


class DetailedTreeNode(TreeNode):
    """Detailed tree node with current period, previous period, diff, pct diff, and contribution.

    Required data keys:
        header: str - Node header text
        percent: float - Percentage change value
        current_period: str - Current period value text
        previous_period: str - Previous period value text
        diff: str - Absolute difference text
        contribution: str - Contribution value text

    Optional data keys:
        current_label: str - Label for current period (default: "Current Period")
        previous_label: str - Label for previous period (default: "Previous Period")
    """

    NODE_WIDTH = 3.0
    NODE_HEIGHT = 1.9

    @staticmethod
    def _get_color(percent_change: float) -> str:
        """Return color based on percent change thresholds.

        Green if > 1%, Red if < -1%, Grey if between -1% and 1%.

        Args:
            percent_change: Percentage change value.

        Returns:
            Hex color code as string.

        """
        if percent_change > 1:
            return COLORS["green"][500]
        if percent_change < -1:
            return COLORS["red"][500]
        return COLORS["gray"][500]

    def render(self, ax: Axes) -> None:
        """Render the detailed node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        # Extract data from the data dict
        header = self._data["header"]
        percent = self._data["percent"]
        current_period = self._data["current_period"]
        previous_period = self._data["previous_period"]
        diff = self._data["diff"]
        contribution = self._data["contribution"]
        # Styling constants
        corner_radius = 0.15
        header_height_ratio = 0.25
        header_color = self._get_color(percent)
        header_text_color = "white"
        data_bg_color = "white"
        data_text_color = COLORS["gray"][600]  # Dark gray for better readability
        label_color = COLORS["gray"][500]  # Medium gray for labels
        border_color = COLORS["gray"][200]  # Light gray for borders

        styling_context = get_styling_context()

        # Get standard font properties
        semi_bold_font = styling_context.get_font_properties(styling_context.fonts.title_font)
        regular_font = styling_context.get_font_properties(styling_context.fonts.label_font)

        # Title section (colored header)
        title_section_height = self.NODE_HEIGHT * header_height_ratio
        data_section_height = self.NODE_HEIGHT - title_section_height

        # Layout configuration based on visual structure (all values are fixed/absolute)
        layout = {
            "node": {
                "padding_left": 0.225,
                "padding_right": 0.225,
            },
            "data_section": {
                "padding_top": 0.21,  # Space above period labels
                "padding_bottom": 0.14,  # Space below last info row
                "period_subsection": {
                    "previous_period_x": 1.65,  # X position for "Previous Period" column
                    "label_row_gap": 0.21,  # Vertical gap from labels to metrics
                    "bottom_margin": 0.21,  # Space below period metrics before divider
                },
                "divider": {
                    "horizontal_inset": 0.225,  # Padding on left/right of divider line
                    "vertical_offset": 0.05,  # Fine-tune vertical position
                },
                "info_subsection": {
                    "row_count": 3,  # Number of info rows
                    "row_spacing_scale": 0.75,  # Scale factor for vertical spacing between rows
                },
            },
        }
        # Render title section box
        title_box = BaseRoundedBox(
            xy=(self.x, self.y + self.NODE_HEIGHT - title_section_height),
            width=self.NODE_WIDTH,
            height=title_section_height,
            top_radius=corner_radius,
            bottom_radius=0,
            facecolor=header_color,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(title_box)

        # Render data section box
        data_section_box = BaseRoundedBox(
            xy=(self.x, self.y),
            width=self.NODE_WIDTH,
            height=data_section_height,
            top_radius=0,
            bottom_radius=corner_radius,
            facecolor=data_bg_color,
            edgecolor=border_color,
            linewidth=1,
        )
        ax.add_patch(data_section_box)

        # Calculate horizontal positions
        content_left_x = self.x + layout["node"]["padding_left"]
        previous_period_x = self.x + layout["data_section"]["period_subsection"]["previous_period_x"]
        content_right_x = self.x + self.NODE_WIDTH - layout["node"]["padding_right"]

        # Render title text
        ax.text(
            content_left_x,
            self.y + self.NODE_HEIGHT - title_section_height / 2,
            header,
            ha="left",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=styling_context.fonts.label_size,
            color=header_text_color,
        )

        # Calculate vertical positions for data section
        data_section_top = self.y + data_section_height

        # Draw horizontal line separator between title and data sections
        title_data_separator_y = self.y + data_section_height
        ax.plot(
            [self.x, self.x + self.NODE_WIDTH],
            [title_data_separator_y, title_data_separator_y],
            color=border_color,
            linewidth=1.5,
            zorder=10,
        )

        # Period subsection: labels and metrics
        period_labels_y = data_section_top - layout["data_section"]["padding_top"]
        period_metrics_y = period_labels_y - layout["data_section"]["period_subsection"]["label_row_gap"]

        # Divider line between period and info subsections
        divider_y = (
            period_metrics_y
            - layout["data_section"]["period_subsection"]["bottom_margin"]
            + layout["data_section"]["divider"]["vertical_offset"]
        )
        divider_inset = layout["data_section"]["divider"]["horizontal_inset"]
        ax.plot(
            [self.x + divider_inset, self.x + self.NODE_WIDTH - divider_inset],
            [divider_y, divider_y],
            color=border_color,
            linewidth=1,
            zorder=10,
        )

        # Get period labels (with defaults)
        current_label = self._data.get("current_label", "Current Period")
        previous_label = self._data.get("previous_label", "Previous Period")

        # Render period subsection (labels and metrics for current and previous periods)
        period_columns = [
            (current_label, current_period, content_left_x),
            (previous_label, previous_period, previous_period_x),
        ]

        for label, metric, x_pos in period_columns:
            # Period label
            ax.text(
                x_pos,
                period_labels_y,
                label,
                ha="left",
                va="center",
                fontproperties=regular_font,
                fontsize=styling_context.fonts.tick_size,
                color=label_color,
            )
            # Period metric
            ax.text(
                x_pos,
                period_metrics_y,
                metric,
                ha="left",
                va="center",
                fontproperties=semi_bold_font,
                fontsize=styling_context.fonts.label_size,
                color=data_text_color,
            )

        # Calculate remaining height for info subsection
        info_section_height = (
            data_section_height
            - layout["data_section"]["padding_top"]
            - layout["data_section"]["period_subsection"]["label_row_gap"]
            - layout["data_section"]["period_subsection"]["bottom_margin"]
            - layout["data_section"]["padding_bottom"]
        )
        info_row_height = info_section_height / layout["data_section"]["info_subsection"]["row_count"]

        # Render info subsection rows (Diff, Pct Diff, Contribution)
        info_rows = [
            ("Diff", diff),
            ("Pct Diff", f"{percent:.1f}%"),
            ("Contribution", contribution),
        ]

        for i, (label, metric) in enumerate(info_rows):
            info_row_y = divider_y - info_row_height * (
                i + layout["data_section"]["info_subsection"]["row_spacing_scale"]
            )
            # Label on left
            ax.text(
                content_left_x,
                info_row_y,
                label,
                ha="left",
                va="center",
                fontproperties=regular_font,
                fontsize=styling_context.fonts.tick_size,
                color=label_color,
            )
            # Metric on right
            ax.text(
                content_right_x,
                info_row_y,
                metric,
                ha="right",
                va="center",
                fontproperties=semi_bold_font,
                fontsize=styling_context.fonts.tick_size,
                color=data_text_color,
            )


class TreeGrid:
    """Grid-based tree diagram renderer with configurable node types."""

    def __init__(
        self,
        tree_structure: dict[str, dict],
        num_rows: int,
        num_cols: int,
        node_class: type[TreeNode],
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
    ) -> None:
        """Initialize the tree grid.

        Args:
            tree_structure: Dictionary mapping node IDs to node data with required keys
                depending on the node_class being used.
            num_rows: Number of rows in the grid.
            num_cols: Number of columns in the grid.
            node_class: The TreeNode subclass to use for rendering nodes.
            vertical_spacing: Vertical spacing between rows. If None, automatically calculated as
                node_height + 0.6 gap.
            horizontal_spacing: Horizontal spacing between columns. If None, automatically calculated as
                node_width - 1.0 overlap for compact layout.
            node_class: Class to use for rendering nodes (must implement TreeNode interface).
            base_x: Starting X position for the leftmost column.
            base_y: Starting Y position for the bottom row.

        """
        self.tree_structure = tree_structure
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.node_class = node_class

        # Get node dimensions from the node class
        self.node_width = node_class.NODE_WIDTH
        self.node_height = node_class.NODE_HEIGHT

        # Auto-calculate spacing if not provided
        self.vertical_spacing = vertical_spacing if vertical_spacing is not None else self.node_height + 0.6
        self.horizontal_spacing = horizontal_spacing if horizontal_spacing is not None else self.node_width - 1.0

        # Generate row and column positions
        self.row = {i: i * self.vertical_spacing for i in range(num_rows)}
        self.col = {i: i * self.horizontal_spacing for i in range(num_cols)}

    def render(self, ax: Axes | None = None) -> tuple[Figure | None, Axes]:
        """Render the tree diagram.

        Args:
            ax: Optional matplotlib axes object. If None, creates a new figure and axes.

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.

        """
        fig = None
        if ax is None:
            # Calculate plot dimensions based on layout
            plot_width = self.col[self.num_cols - 1] + self.node_width
            plot_height = self.row[self.num_rows - 1] + self.node_height

            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlim(0, plot_width)
            ax.set_ylim(0, plot_height)
            ax.axis("off")

        # First pass: create all nodes and store their centers
        node_centers = {}
        for node_id, node_data in self.tree_structure.items():
            # Convert grid coordinates to absolute positions
            col_idx, row_idx = node_data["position"]
            x = self.col[col_idx]
            y = self.row[row_idx]

            # Extract data for the node (exclude position and children which are structural)
            data_dict = {k: v for k, v in node_data.items() if k not in ("position", "children")}

            # Create and render the node
            node = self.node_class(
                x=x,
                y=y,
                data=data_dict,
            )
            node.render(ax)

            # Store center positions for connections
            node_centers[node_id] = {
                "x": x + self.node_width / 2,
                "y_bottom": y,
                "y_top": y + self.node_height,
            }

        # Second pass: draw connections
        for node_id, node_data in self.tree_structure.items():
            if "children" in node_data and len(node_data["children"]) > 0:
                parent = node_centers[node_id]
                for child_id in node_data["children"]:
                    child = node_centers[child_id]
                    self._draw_connection(
                        ax=ax,
                        x1=parent["x"],
                        y1=parent["y_bottom"],
                        x2=child["x"],
                        y2=child["y_top"],
                    )

        return fig, ax

    @staticmethod
    def _add_curve(
        verts: list[tuple[float, float]],
        codes: list[int],
        x: float,
        y: float,
        x_offset: float,
        y_offset: float,
    ) -> None:
        """Add Bezier curve control points to the path.

        Args:
            verts: List of vertices to append to.
            codes: List of path codes to append to.
            x: X-coordinate of the curve start point.
            y: Y-coordinate of the curve start point.
            x_offset: X offset for the curve end point.
            y_offset: Y offset for the curve end point.

        """
        verts.append((x, y))
        codes.append(Path.CURVE3)
        verts.append((x + x_offset, y + y_offset))
        codes.append(Path.CURVE3)

    @staticmethod
    def _draw_connection(ax: Axes, x1: float, y1: float, x2: float, y2: float) -> None:
        """Draw connection line between nodes with curved corners.

        Args:
            ax: Matplotlib axes object.
            x1: X-coordinate of first point.
            y1: Y-coordinate of first point.
            x2: X-coordinate of second point.
            y2: Y-coordinate of second point.

        """
        # Connection styling constants
        curve_radius = 0.15
        line_width = 2
        line_color = "black"

        mid_y = (y1 + y2) / 2
        curve_sign = 1 if x2 > x1 else -1

        # Create path with curved corners using Bezier curves
        verts = []
        codes = []

        # Start point (bottom of parent node)
        verts.append((x1, y1))
        codes.append(Path.MOVETO)

        # Vertical line down to curve start
        verts.append((x1, mid_y + curve_radius))
        codes.append(Path.LINETO)

        # Curve from vertical to horizontal (first corner)
        TreeGrid._add_curve(verts, codes, x1, mid_y, curve_sign * curve_radius, 0)

        # Horizontal line
        verts.append((x2 - (curve_sign * curve_radius), mid_y))
        codes.append(Path.LINETO)

        # Curve from horizontal to vertical (second corner)
        TreeGrid._add_curve(verts, codes, x2, mid_y, 0, -curve_radius)

        # Vertical line up to child node
        verts.append((x2, y2))
        codes.append(Path.LINETO)

        # Create and draw the path
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=line_color, linewidth=line_width)
        ax.add_patch(patch)


def create_revenue_tree() -> tuple[Figure, Axes]:
    """Create the revenue tree diagram.

    Returns:
        Tuple of (figure, axes) objects.

    """
    # Define tree structure with grid coordinates (col, row)
    tree_structure = {
        "total_sales": {
            "header": "£££ Total Sales (TISP)",
            "percent": -31.5,
            "value1": "£889.5k",
            "value2": "£1.3m",
            "position": (1, 4),
            "children": ["non_card_sales", "ad_card_sales"],
        },
        "non_card_sales": {
            "header": "£££ Non-Card Sales (TISP)",
            "percent": -37.4,
            "value1": "£241.7k",
            "value2": "£385.8k",
            "position": (0, 3),
            "children": [],
        },
        "ad_card_sales": {
            "header": "£££ Ad Card Sales (TISP)",
            "percent": 15.0,
            "value1": "£647.8k",
            "value2": "£912.7k",
            "position": (2, 3),
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
            "position": (2, 1),
            "children": [],
        },
        "av_transaction_value": {
            "header": "Av. Transaction Value",
            "percent": -8.5,
            "value1": "£8.44",
            "value2": "£9.22",
            "position": (4, 1),
            "children": ["items_per_transaction", "av_item_value"],
        },
        "items_per_transaction": {
            "header": "Items Per Transaction",
            "percent": -1.0,
            "value1": "1.55",
            "value2": "1.56",
            "position": (3, 0),
            "children": [],
        },
        "av_item_value": {
            "header": "Av. Item Value",
            "percent": -7.6,
            "value1": "£5.45",
            "value2": "£5.90",
            "position": (5, 0),
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

    fig, ax = tree_grid.render()

    plt.tight_layout()
    plt.savefig("tree_diagram_matplotlib.png", dpi=300)

    return fig, ax


def create_detailed_revenue_tree() -> tuple[Figure, Axes]:
    """Create the revenue tree diagram with detailed nodes.

    Returns:
        Tuple of (figure, axes) objects.

    """
    # Define tree structure with detailed information
    tree_structure = {
        "revenue": {
            "header": "Revenue",
            "percent": 21.5,
            "current_period": "15,705.00",
            "previous_period": "12,922.92",
            "diff": "2,782.08",
            "contribution": "8.68M",
            "position": (1, 3),
            "children": ["customers", "spend_per_customer"],
        },
        "customers": {
            "header": "Customers",
            "percent": 8.5,
            "current_period": "4.12",
            "previous_period": "3.80",
            "diff": "0.32",
            "contribution": "4.23M",
            "position": (0, 2),
            "children": [],
        },
        "spend_per_customer": {
            "header": "Spend / Customer",
            "percent": 12.1,
            "current_period": "3,810.92",
            "previous_period": "3,400.76",
            "diff": "410.16",
            "contribution": "4.45M",
            "position": (2, 2),
            "children": ["visits_per_customer", "spend_per_visit"],
        },
        "visits_per_customer": {
            "header": "Visits / Customer",
            "percent": -5.2,
            "current_period": "12.4",
            "previous_period": "13.1",
            "diff": "-0.7",
            "contribution": "-1.82M",
            "position": (1, 1),
            "children": [],
        },
        "spend_per_visit": {
            "header": "Spend / Visit",
            "percent": 18.3,
            "current_period": "307.33",
            "previous_period": "259.68",
            "diff": "47.65",
            "contribution": "6.27M",
            "position": (3, 1),
            "children": ["units_per_visit", "price_per_unit"],
        },
        "units_per_visit": {
            "header": "Units / Visit",
            "percent": 15.8,
            "current_period": "285.12",
            "previous_period": "246.22",
            "diff": "38.90",
            "contribution": "5.12M",
            "position": (2, 0),
            "children": [],
        },
        "price_per_unit": {
            "header": "Price / Unit",
            "percent": 0.5,
            "current_period": "22.21",
            "previous_period": "21.73",
            "diff": "0.48",
            "contribution": "1.15M",
            "position": (4, 0),
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

    fig, ax = tree_grid.render()

    plt.tight_layout()
    plt.savefig("tree_diagram_detailed.png", dpi=300)

    return fig, ax


if __name__ == "__main__":
    # Create both trees
    create_revenue_tree()
    create_detailed_revenue_tree()
