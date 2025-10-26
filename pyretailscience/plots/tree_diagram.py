"""Tree Diagram Module.

This module implements a TreeDiagram class for creating hierarchical tree visualizations
using Graphviz. The TreeDiagram class can be used by various analysis classes to create
reusable tree-based visualizations.
"""

import subprocess
from textwrap import dedent
from typing import Any

import graphviz
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.path import Path

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
