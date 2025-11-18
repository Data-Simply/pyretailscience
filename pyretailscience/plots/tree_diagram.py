"""Tree Diagram Module.

This module implements tree visualization components using matplotlib for creating
hierarchical tree diagrams with custom node types and grid-based layouts.
"""

from abc import ABC, abstractmethod
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path

from pyretailscience.plots.styles.styling_context import get_styling_context
from pyretailscience.plots.styles.tailwind import COLORS


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

    NODE_WIDTH = 3.75
    NODE_HEIGHT = 1.75

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
        value_vertical_offset = 0.175

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

    # Connection styling constants
    CONNECTION_CURVE_RADIUS = 0.15
    CONNECTION_LINE_WIDTH = 2
    CONNECTION_LINE_COLOR = "black"

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

        Raises:
            ValueError: If grid dimensions are not positive, if tree_structure is empty,
                or if node positions are out of bounds.
            TypeError: If node_class is not a TreeNode subclass.

        """
        # Validate grid dimensions
        if num_rows <= 0 or num_cols <= 0:
            error_msg = f"Grid dimensions must be positive: num_rows={num_rows}, num_cols={num_cols}"
            raise ValueError(error_msg)

        # Validate node_class is a TreeNode subclass
        if not issubclass(node_class, TreeNode):
            error_msg = f"node_class must be a TreeNode subclass, got {node_class}"
            raise TypeError(error_msg)

        # Validate tree_structure is not empty
        if not tree_structure:
            raise ValueError("tree_structure cannot be empty")

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

        # Validate positions are within grid bounds
        for node_id, node_data in tree_structure.items():
            if "position" not in node_data:
                error_msg = f"Node '{node_id}' is missing required 'position' key"
                raise ValueError(error_msg)

            col_idx, row_idx = node_data["position"]
            if not (0 <= col_idx < num_cols):
                error_msg = f"Node '{node_id}' column index {col_idx} is out of bounds [0, {num_cols})"
                raise ValueError(error_msg)
            if not (0 <= row_idx < num_rows):
                error_msg = f"Node '{node_id}' row index {row_idx} is out of bounds [0, {num_rows})"
                raise ValueError(error_msg)

        # Generate row and column positions
        # Row 0 is at the top, increasing downward (reversed from matplotlib's default bottom-up)
        self.row = {i: (num_rows - 1 - i) * self.vertical_spacing for i in range(num_rows)}
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
            # Row 0 is at the top with the highest y-value after coordinate inversion
            plot_height = self.row[0] + self.node_height

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
                    if child_id not in node_centers:
                        available_nodes = list(node_centers.keys())
                        error_msg = (
                            f"Child node '{child_id}' referenced by '{node_id}' not found in tree_structure. "
                            f"Available nodes: {available_nodes}"
                        )
                        raise ValueError(error_msg)
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
        # Use class constants for connection styling
        curve_radius = TreeGrid.CONNECTION_CURVE_RADIUS
        line_width = TreeGrid.CONNECTION_LINE_WIDTH
        line_color = TreeGrid.CONNECTION_LINE_COLOR

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


class DetailedTreeNode(TreeNode):
    """Detailed tree node with current period, previous period, diff, pct diff, and contribution.

    Required data keys:
        header: str - Node header text
        percent: float - Percentage change value
        current_period: str - Current period value text
        previous_period: str - Previous period value text
        diff: str - Absolute difference text

    Optional data keys:
        contribution: str - Contribution value text (if not provided, row is left blank)
        current_label: str - Label for current period (default: "Current Period")
        previous_label: str - Label for previous period (default: "Previous Period")
    """

    NODE_WIDTH = 3.5
    NODE_HEIGHT = 2.5

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
        if percent_change >= DetailedTreeNode.GREEN_THRESHOLD:
            return COLORS["green"][500]
        if percent_change <= DetailedTreeNode.RED_THRESHOLD:
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
        contribution = self._data.get("contribution")
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
                "padding_left": 0.2,
                "padding_right": 0.2,
            },
            "data_section": {
                "padding_top": 0.2,  # Space above period labels
                "padding_bottom": 0.1,  # Space below last info row
                "period_subsection": {
                    "previous_period_x": (self.NODE_WIDTH / 2) + 0.2,  # X position for "Previous Period" column
                    "label_row_gap": 0.25,  # Vertical gap from labels to metrics
                    "bottom_margin": 0.25,  # Space below period metrics before divider
                },
                "divider": {
                    "horizontal_inset": 0.2,  # Padding on left/right of divider line
                    "vertical_offset": 0.025,  # Fine-tune vertical position
                },
                "info_subsection": {
                    "row_count": 3,  # Number of info rows
                    "row_spacing_scale": 0.75,  # Scale factor for vertical spacing between rows
                },
            },
        }
        # Render title section box (border matches header color for seamless appearance)
        title_box = BaseRoundedBox(
            xy=(self.x, self.y + self.NODE_HEIGHT - title_section_height),
            width=self.NODE_WIDTH,
            height=title_section_height,
            top_radius=corner_radius,
            bottom_radius=0,
            facecolor=header_color,
            edgecolor=header_color,
            linewidth=1,
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
        ]
        # Only add contribution row if it's provided
        if contribution is not None:
            info_rows.append(("Contribution", contribution))

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


class LightGBMTreeNode(TreeNode):
    """Tree node for visualizing LightGBM decision tree nodes.

    Displays decision tree information including split conditions, leaf values,
    sample counts, and average values. Uses a two-section layout with colored
    header and content area.
    """

    NODE_WIDTH = 3.5
    NODE_HEIGHT = 1.7
    HEADER_HEIGHT_FRACTION = 0.25

    # Default dimensions used for responsive font scaling
    DEFAULT_WIDTH = 3.5
    DEFAULT_HEIGHT = 1.7

    def render(self, ax: Axes) -> None:
        """Render the LightGBM tree node on the given axes.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes object to render on.
        """
        data = self._data
        header_height = self.NODE_HEIGHT * self.HEADER_HEIGHT_FRACTION
        content_height = self.NODE_HEIGHT - header_height

        # Determine node color based on value (red -> yellow -> green)
        node_value = data.get("value", 0)
        value_range = data.get("value_range", (node_value, node_value))
        min_val, max_val = value_range

        # Normalize value to 0-1 range
        normalized = (node_value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        # Color gradient: red (0) -> yellow (0.5) -> green (1)
        mid_point = 0.5
        color_threshold_low = 0.33
        color_threshold_high = 0.67

        if normalized < mid_point:
            # Red to yellow
            t = normalized * 2
            color = (
                COLORS["red"][500]
                if t < color_threshold_low
                else COLORS["orange"][500]
                if t < color_threshold_high
                else COLORS["yellow"][500]
            )
        else:
            # Yellow to green
            t = (normalized - mid_point) * 2
            color = (
                COLORS["yellow"][500]
                if t < color_threshold_low
                else COLORS["lime"][500]
                if t < color_threshold_high
                else COLORS["green"][500]
            )

        # Draw header box with colored background
        header_box = BaseRoundedBox(
            xy=(self.x, self.y + content_height),
            width=self.NODE_WIDTH,
            height=header_height,
            top_radius=0.15,
            bottom_radius=0,
            facecolor=color,
            edgecolor=COLORS["gray"][700],
            linewidth=1.5,
        )
        ax.add_patch(header_box)

        # Draw content box
        content_box = BaseRoundedBox(
            xy=(self.x, self.y),
            width=self.NODE_WIDTH,
            height=content_height,
            top_radius=0,
            bottom_radius=0.15,
            facecolor="white",
            edgecolor=COLORS["gray"][700],
            linewidth=1.5,
        )
        ax.add_patch(content_box)

        # Get responsive font sizes
        font_sizes = self._get_responsive_font_sizes()

        # Header text (feature name or "Leaf")
        header_text = data.get("split_feature", "Leaf")
        ax.text(
            self.x + self.NODE_WIDTH / 2,
            self.y + content_height + header_height / 2,
            header_text,
            ha="center",
            va="center",
            fontsize=font_sizes["header_size"],
            fontweight="bold",
            color="white",
        )

        # Content area text
        y_offset = self.y + content_height * 0.85
        line_height = content_height * 0.18

        # Split condition or leaf value
        if "split_condition" in data:
            condition_text = data["split_condition"]
            ax.text(
                self.x + self.NODE_WIDTH / 2,
                y_offset,
                condition_text,
                ha="center",
                va="center",
                fontsize=font_sizes["content_size"],
                color=COLORS["gray"][900],
            )
            y_offset -= line_height

        # Sample count
        if "sample_count" in data:
            count_text = f"Samples: {data['sample_count']:,}"
            ax.text(
                self.x + self.NODE_WIDTH * 0.1,
                y_offset,
                count_text,
                ha="left",
                va="center",
                fontsize=font_sizes["label_size"],
                color=COLORS["gray"][700],
            )
            y_offset -= line_height

        # Average value
        if "avg_value" in data:
            avg_text = (
                f"Avg: {data['avg_value']:.2%}" if abs(data["avg_value"]) < 1 else f"Avg: {data['avg_value']:.2f}"
            )
            ax.text(
                self.x + self.NODE_WIDTH * 0.1,
                y_offset,
                avg_text,
                ha="left",
                va="center",
                fontsize=font_sizes["label_size"],
                color=COLORS["gray"][700],
            )

    def _get_responsive_font_sizes(self) -> dict[str, float]:
        """Calculate responsive font sizes based on node dimensions.

        Scales fonts relative to the default dimensions to maintain readability
        when node sizes are customized.

        Returns:
            dict[str, float]: Dictionary with header_size, content_size, and label_size.
        """
        width_scale = self.NODE_WIDTH / self.DEFAULT_WIDTH
        height_scale = self.NODE_HEIGHT / self.DEFAULT_HEIGHT
        scale_factor = min(width_scale, height_scale)

        return {
            "header_size": 12 * scale_factor,
            "content_size": 11 * scale_factor,
            "label_size": 9 * scale_factor,
        }

    def get_width(self) -> float:
        """Return the node width.

        Returns:
            float: Node width in data coordinates.
        """
        return self.NODE_WIDTH

    def get_height(self) -> float:
        """Return the node height.

        Returns:
            float: Node height in data coordinates.
        """
        return self.NODE_HEIGHT


def lightgbm_tree_to_grid(  # noqa: C901
    booster: Any,  # noqa: ANN401
    feature_names: list[str] | None = None,
) -> tuple[dict[str, dict], int, int]:
    """Convert LightGBM single-tree booster to TreeGrid format.

    Parses the LightGBM tree structure and creates a dictionary compatible with TreeGrid.
    Automatically calculates value ranges for color normalization and tree dimensions.

    Args:
        booster: LightGBM Booster object (must have single tree, n_estimators=1).
        feature_names (list[str] | None): Optional list of feature names for display. If None, uses split indices.

    Returns:
        tuple[dict[str, dict], int, int]: Tuple of (tree_structure, num_rows, num_cols) where:
            - tree_structure: Dict mapping node_id -> node data with children list
            - num_rows: Number of levels in the tree (tree depth + 1)
            - num_cols: Maximum width of the tree at any level

    Raises:
        ValueError: If booster doesn't have exactly one tree.

    Example:
        >>> from lightgbm import LGBMRegressor
        >>> model = LGBMRegressor(n_estimators=1, max_depth=3)
        >>> model.fit(X_train, y_train)
        >>> tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(
        ...     model.booster_, feature_names=X_train.columns
        ... )
        >>> grid = TreeGrid(
        ...     tree_structure=tree_structure,
        ...     num_rows=num_rows,
        ...     num_cols=num_cols,
        ...     node_class=LightGBMTreeNode
        ... )
        >>> grid.render()
    """
    import numpy as np

    # Get tree structure from booster
    tree_dump = booster.dump_model()
    tree_info_list = tree_dump["tree_info"]

    if len(tree_info_list) != 1:
        msg = f"Expected single tree (n_estimators=1), got {len(tree_info_list)} trees"
        raise ValueError(msg)

    tree_structure_raw = tree_info_list[0]["tree_structure"]

    # Collect all node values for normalization
    all_values: list[float] = []

    def collect_values(node: dict[str, Any]) -> None:
        if "leaf_value" in node:
            all_values.append(node["leaf_value"])
        elif "internal_value" in node:
            all_values.append(node["internal_value"])
        if "left_child" in node:
            collect_values(node["left_child"])
        if "right_child" in node:
            collect_values(node["right_child"])

    collect_values(tree_structure_raw)

    if not all_values:
        all_values = [0]

    value_range = (float(np.min(all_values)), float(np.max(all_values)))

    # Convert tree structure to TreeGrid format and track positions
    tree_structure: dict[str, dict] = {}
    node_counter = [0]  # Use list for mutable counter
    node_positions: dict[str, tuple[int, int]] = {}  # node_id -> (row, col)

    def convert_node(lgbm_node: dict[str, Any], row: int, col: int) -> str:
        """Recursively convert LightGBM node to TreeGrid format."""
        node_id = f"node_{node_counter[0]}"
        node_counter[0] += 1

        # Track position
        node_positions[node_id] = (row, col)

        # Extract node data
        is_leaf = "leaf_value" in lgbm_node
        node_value = lgbm_node.get("leaf_value", lgbm_node.get("internal_value", 0))
        sample_count = lgbm_node.get("leaf_count", lgbm_node.get("internal_count", 0))

        # Build node data dict
        node_data = {
            "value": float(node_value),
            "sample_count": int(sample_count),
            "avg_value": float(node_value),
            "value_range": value_range,
            "position": (col, row),  # TreeGrid expects (col_idx, row_idx) tuple
            "children": [],
        }

        if is_leaf:
            # Leaf node
            node_data["split_feature"] = "Leaf"
        else:
            # Internal node with split
            split_feature_idx = lgbm_node.get("split_feature")
            threshold = lgbm_node.get("threshold")
            decision_type = lgbm_node.get("decision_type", "<=")

            # Get feature name
            if feature_names and split_feature_idx < len(feature_names):
                feature_name = feature_names[split_feature_idx]
            else:
                feature_name = f"Feature {split_feature_idx}"

            node_data["split_feature"] = feature_name
            node_data["split_condition"] = f"{decision_type} {threshold:.4f}" if threshold is not None else ""

            # Process children (left goes to next row, col*2; right goes to next row, col*2+1)
            if "left_child" in lgbm_node:
                left_child_id = convert_node(lgbm_node["left_child"], row + 1, col * 2)
                node_data["children"].append(left_child_id)

            if "right_child" in lgbm_node:
                right_child_id = convert_node(lgbm_node["right_child"], row + 1, col * 2 + 1)
                node_data["children"].append(right_child_id)

        tree_structure[node_id] = node_data
        return node_id

    # Convert starting from root at row=0, col=0
    convert_node(tree_structure_raw, row=0, col=0)

    # Calculate tree dimensions
    max_row = max(row for row, _ in node_positions.values())
    max_col = max(col for _, col in node_positions.values())

    num_rows = max_row + 1
    num_cols = max_col + 1

    return tree_structure, num_rows, num_cols
