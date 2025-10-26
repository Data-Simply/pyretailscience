"""Tree Diagram Module.

This module implements a TreeDiagram class for creating hierarchical tree visualizations
using Graphviz. The TreeDiagram class can be used by various analysis classes to create
reusable tree-based visualizations.
"""

import subprocess
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any

import graphviz
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path

from pyretailscience.plots.styles import graph_utils as gu
from pyretailscience.plots.styles.styling_context import get_styling_context
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
        try:
            subprocess.run(["dot", "-V"], check=True, stderr=subprocess.DEVNULL)  # noqa: S603 S607
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


class BaseRoundedBox(mpatches.PathPatch):
    """Base patch with independent corner rounding for top and bottom."""

    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        top_radius: float = 0.3,
        bottom_radius: float = 0.3,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the custom rounded box.

        Args:
            xy: Bottom-left corner coordinates (x, y).
            width: Width of the box.
            height: Height of the box.
            top_radius: Radius for top corners.
            bottom_radius: Radius for bottom corners.
            **kwargs: Additional keyword arguments for PathPatch.

        """
        x, y = xy

        # Define the path with different corner radii
        verts = []
        codes = []

        # Define corner configurations: (center_x, center_y, radius, start_angle, end_angle, fallback_point, is_first)
        corners = [
            # Bottom left corner
            (x + bottom_radius, y + bottom_radius, bottom_radius, np.pi, 3 * np.pi / 2, (x, y), True),
            # Bottom right corner
            (
                x + width - bottom_radius,
                y + bottom_radius,
                bottom_radius,
                3 * np.pi / 2,
                2 * np.pi,
                (x + width, y),
                False,
            ),
            # Top right corner
            (x + width - top_radius, y + height - top_radius, top_radius, 0, np.pi / 2, (x + width, y + height), False),
            # Top left corner
            (x + top_radius, y + height - top_radius, top_radius, np.pi / 2, np.pi, (x, y + height), False),
        ]

        # Generate all corners
        for center_x, center_y, radius, start_angle, end_angle, fallback_point, is_first in corners:
            corner_verts, corner_codes = self._generate_corner(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                fallback_point=fallback_point,
                is_first=is_first,
            )
            verts.extend(corner_verts)
            codes.extend(corner_codes)

        codes.append(Path.CLOSEPOLY)
        verts.append((0, 0))

        path = Path(verts, codes)
        super().__init__(path, **kwargs)

    @staticmethod
    def _generate_corner(
        center_x: float,
        center_y: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        fallback_point: tuple[float, float],
        is_first: bool,
    ) -> tuple[list[tuple[float, float]], list[int]]:
        """Generate vertices and codes for a single corner.

        Args:
            center_x: X-coordinate of the corner's center.
            center_y: Y-coordinate of the corner's center.
            radius: Radius of the corner.
            start_angle: Starting angle for the corner arc (in radians).
            end_angle: Ending angle for the corner arc (in radians).
            fallback_point: Point to use if radius is 0.
            is_first: Whether this is the first corner (uses MOVETO instead of LINETO).

        Returns:
            Tuple of (vertices, codes) for this corner.

        """
        arc_points = 10  # Number of points to use for the corner arc

        if radius > 0:
            theta = np.linspace(start_angle, end_angle, arc_points)

            verts = [(center_x + radius * np.cos(t), center_y + radius * np.sin(t)) for t in theta]
            codes = [Path.MOVETO] + [Path.LINETO] * (arc_points - 1) if is_first else [Path.LINETO] * arc_points

            return verts, codes

        verts = [fallback_point]
        codes = [Path.MOVETO if is_first else Path.LINETO]

        return verts, codes


class TreeNode(ABC):
    """Abstract base class for tree nodes.

    All TreeNode subclasses must define NODE_WIDTH and NODE_HEIGHT class attributes
    and implement the render() method.
    """

    # Subclasses must define these class attributes
    NODE_WIDTH: float
    NODE_HEIGHT: float

    def __init__(
        self,
        data: dict[str, Any],
        x: float,
        y: float,
    ) -> None:
        """Initialize the tree node.

        Args:
            data: Dictionary containing node data. Each subclass defines required keys.
            x: X-coordinate of bottom-left corner.
            y: Y-coordinate of bottom-left corner.

        """
        self._data = data
        self.x = x
        self.y = y

    @abstractmethod
    def render(self, ax: Axes) -> None:
        """Render the node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.

        """
        ...

    def get_width(self) -> float:
        """Return the node width.

        Returns:
            float: Node width.

        """
        return self.NODE_WIDTH

    def get_height(self) -> float:
        """Return the node height.

        Returns:
            float: Node height.

        """
        return self.NODE_HEIGHT


class SimpleTreeNode(TreeNode):
    """Simple tree node implementation with header and data sections.

    Required data keys:
        header (str): The header text
        percent (float): Percentage change value
        value1 (str): First value text
        value2 (str): Second value text
    """

    NODE_WIDTH = 3.0
    NODE_HEIGHT = 1.2

    # Color thresholds for percent change
    GREEN_THRESHOLD = 1.0  # Percent change at or above this shows green
    RED_THRESHOLD = -1.0  # Percent change at or below this shows red

    @staticmethod
    def _get_color(percent_change: float) -> str:
        """Return color based on percent change thresholds.

        Green if >= GREEN_THRESHOLD, Red if <= RED_THRESHOLD, Grey otherwise.

        Args:
            percent_change: Percentage change value.

        Returns:
            str: Hex color code as string.

        """
        if percent_change >= SimpleTreeNode.GREEN_THRESHOLD:
            return COLORS["green"][500]
        if percent_change <= SimpleTreeNode.RED_THRESHOLD:
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

    def render(self, ax: Axes | None = None) -> Axes:
        """Render the tree diagram.

        Args:
            ax: Optional matplotlib axes object. If None, creates a new figure and axes.

        Returns:
            The matplotlib axes object.

        """
        if ax is None:
            # Calculate plot dimensions based on layout
            plot_width = self.col[self.num_cols - 1] + self.node_width
            plot_height = self.row[self.num_rows - 1] + self.node_height

            _, ax = plt.subplots(figsize=(plot_width, plot_height))
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
                data=data_dict,
                x=x,
                y=y,
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

        return ax

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
