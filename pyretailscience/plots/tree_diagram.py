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

    def _get_responsive_font_sizes(self, styling_context: Any) -> dict[str, float]:  # noqa: ANN401
        """Calculate responsive font sizes based on node dimensions.

        Scales font sizes proportionally to node dimensions relative to defaults.
        Uses the smaller of width/height scaling to ensure text fits.

        Args:
            styling_context: Styling context containing default font sizes.

        Returns:
            dict[str, float]: Dictionary with 'label_size' and 'title_size' keys.

        """
        # Default dimensions (what the fixed font sizes were designed for)
        default_width = 3.75
        default_height = 1.75

        # Calculate scaling factors
        width_scale = self.NODE_WIDTH / default_width
        height_scale = self.NODE_HEIGHT / default_height

        # Use the smaller scale factor to ensure text fits
        scale_factor = min(width_scale, height_scale)

        # Scale the default font sizes
        base_label_size = styling_context.fonts.label_size
        base_title_size = styling_context.fonts.title_size

        return {
            "label_size": base_label_size * scale_factor,
            "title_size": base_title_size * scale_factor,
        }

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

        # Calculate responsive font sizes based on node dimensions
        font_sizes = self._get_responsive_font_sizes(styling_context)

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
            fontsize=font_sizes["label_size"],
            color=text_color,
        )

        ax.text(
            self.x + self.NODE_WIDTH * percent_text_x_fraction,
            self.y + data_height / 2,
            f"{percent:+.1f}%",
            ha="center",
            va="center",
            fontproperties=semi_bold_font,
            fontsize=font_sizes["title_size"],
            color=text_color,
        )

        ax.text(
            self.x + self.NODE_WIDTH * value_text_x_fraction,
            self.y + data_height / 2 + value_vertical_offset,
            value1,
            ha="left",
            va="center",
            fontproperties=regular_font,
            fontsize=font_sizes["label_size"],
            color=text_color,
        )
        ax.text(
            self.x + self.NODE_WIDTH * value_text_x_fraction,
            self.y + data_height / 2 - value_vertical_offset,
            value2,
            ha="left",
            va="center",
            fontproperties=regular_font,
            fontsize=font_sizes["label_size"],
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
        node_class: type[TreeNode],
        node_width: float | None = None,
        node_height: float | None = None,
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
    ) -> None:
        """Initialize the tree grid with automatic layout.

        Args:
            tree_structure: Dictionary mapping node IDs to node data. Each node must include a
                'children' key (list[str]) listing child node IDs. Use empty list for leaf nodes.
            node_class: The TreeNode subclass to use for rendering nodes.
            node_width: Override the default node width from node_class. Defaults to None (use class default).
            node_height: Override the default node height from node_class. Defaults to None (use class default).
            vertical_spacing: Vertical spacing between rows. If None, automatically calculated as
                node_height + 0.6 gap.
            horizontal_spacing: Horizontal spacing between columns. If None, automatically calculated as
                node_width + 0.5 gap.

        Raises:
            ValueError: If tree_structure is empty.
            TypeError: If node_class is not a TreeNode subclass.

        """
        # Validate node_class is a TreeNode subclass
        if not issubclass(node_class, TreeNode):
            error_msg = f"node_class must be a TreeNode subclass, got {node_class}"
            raise TypeError(error_msg)

        # Validate tree_structure is not empty
        if not tree_structure:
            raise ValueError("tree_structure cannot be empty")

        self.tree_structure = tree_structure
        self.node_class = node_class

        # Get node dimensions from the node class or use overrides
        self.node_width = node_width if node_width is not None else node_class.NODE_WIDTH
        self.node_height = node_height if node_height is not None else node_class.NODE_HEIGHT

        # Auto-calculate spacing if not provided
        self.vertical_spacing = vertical_spacing if vertical_spacing is not None else self.node_height + 0.6
        self.horizontal_spacing = horizontal_spacing if horizontal_spacing is not None else self.node_width + 0.5

        # Always compute positions automatically from tree structure
        positions, computed_rows, computed_cols = self._compute_positions()

        # Store computed positions (we'll use them during rendering)
        self._positions = positions
        self.num_rows = computed_rows
        self.num_cols = computed_cols

        # Generate row and column positions
        # Row 0 is at the top, increasing downward (reversed from matplotlib's default bottom-up)
        self.row = {i: (self.num_rows - 1 - i) * self.vertical_spacing for i in range(self.num_rows)}
        self.col = {i: i * self.horizontal_spacing for i in range(self.num_cols)}

    def render(self, ax: Axes | None = None) -> Axes:
        """Render the tree diagram.

        Args:
            ax: Optional matplotlib axes object. If None, creates a new figure and axes.

        Returns:
            The matplotlib axes object.

        """
        if ax is None:
            # Calculate plot dimensions based on actual node positions
            # Find the maximum x position from all nodes
            max_x = max(
                col_idx * self.horizontal_spacing
                for col_idx, _ in self._positions.values()
            )
            # Add padding for half node width on each side
            plot_width = max_x + self.node_width

            # Row 0 is at the top with the highest y-value after coordinate inversion
            plot_height = self.row[0] + self.node_height

            _, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlim(0, plot_width)
            ax.set_ylim(0, plot_height)
            ax.axis("off")

        # First pass: create all nodes and store their centers
        node_centers = {}
        for node_id, node_data in self.tree_structure.items():
            # Get computed position from auto-layout
            col_idx, row_idx = self._positions[node_id]
            x = col_idx * self.horizontal_spacing  # Calculate x from float column index
            y = self.row[row_idx]

            # Extract data for the node (exclude children which is structural)
            data_dict = {k: v for k, v in node_data.items() if k != "children"}

            # Create and render the node
            node = self.node_class(
                data=data_dict,
                x=x,
                y=y,
            )
            # Override node dimensions if custom dimensions were specified
            node.NODE_WIDTH = self.node_width
            node.NODE_HEIGHT = self.node_height
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

    def _compute_positions(self) -> tuple[dict[str, tuple[int, int]], int, int]:
        """Compute grid positions using centered tree auto-layout.

        Uses a bottom-up algorithm to center parents over their children, then scales
        positions by a uniform spacing factor. This produces a visually balanced tree
        layout with consistent sibling spacing. Layout orientation: top-down (root at top),
        left-to-right (siblings ordered left to right).

        Returns:
            positions: mapping of node_id -> (col, row)
            num_rows: total number of rows required
            num_cols: maximum number of columns across rows
        """
        # Build child adjacency and in-degree to find roots
        children_map: dict[str, list[str]] = {}
        referenced_as_child: set[str] = set()
        for node_id, node_data in self.tree_structure.items():
            children = node_data.get("children", []) or []
            children_map[node_id] = list(children)
            for c in children:
                referenced_as_child.add(c)
        # Root candidates are nodes never referenced as a child
        roots = [nid for nid in self.tree_structure if nid not in referenced_as_child]
        if not roots:
            # Fallback: choose deterministic root
            roots = [sorted(self.tree_structure.keys())[0]]

        # Level-order traversal to assign rows
        from collections import deque

        level_of: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for r in roots:
            queue.append((r, 0))
        visited: set[str] = set()

        while queue:
            nid, lvl = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            level_of[nid] = lvl
            for child in children_map.get(nid, []):
                queue.append((child, lvl + 1))

        # Group nodes by level with stable ordering
        nodes_in_level = self._group_nodes_by_level(level_of)

        # Assign columns using centered layout algorithm
        positions, max_cols = self._assign_columns_centered(nodes_in_level, children_map)

        num_rows = len(nodes_in_level.keys()) if nodes_in_level else 1
        num_cols = max_cols if max_cols > 0 else 1

        return positions, num_rows, num_cols

    def _group_nodes_by_level(self, level_of: dict[str, int]) -> dict[int, list[str]]:
        """Group node ids by their computed level (row), preserving input order per level."""
        from collections import defaultdict

        nodes_in_level: dict[int, list[str]] = defaultdict(list)
        for nid in self.tree_structure:
            lvl = level_of.get(nid, 0)
            nodes_in_level[lvl].append(nid)
        return nodes_in_level

    def _assign_columns_centered(
        self,
        nodes_in_level: dict[int, list[str]],
        children_map: dict[str, list[str]],
    ) -> tuple[dict[str, tuple[int, int]], int]:
        """Assign column positions using bottom-up traversal with uniform spacing.

        Uses a two-pass approach:
        1. Bottom-up: leaves get sequential positions, parents are centered over children
        2. Level-by-level: enforce uniform spacing between siblings, shifting subtrees as needed

        This ensures both aesthetic centering and consistent spacing at all levels.

        Args:
            nodes_in_level: Mapping of level (row) to list of node IDs at that level.
            children_map: Mapping of node ID to list of child node IDs.

        Returns:
            Tuple of (positions dict, max_cols count) where positions maps node_id -> (col, row).
        """
        # Find root nodes
        all_children = {child for children in children_map.values() for child in children}
        roots = [node for node in nodes_in_level.get(0, []) if node not in all_children]

        if not roots:
            roots = list(nodes_in_level.get(0, []))

        positions: dict[str, tuple[int, int]] = {}
        spacing = 2  # Uniform spacing between all siblings
        next_col = [0]  # Use list to allow modification in nested function

        # Pass 1: Layout each root subtree with bottom-up algorithm (centering parents)
        for root in roots:
            self._layout_subtree_uniform(root, 0, children_map, positions, next_col, spacing)

        # Pass 2: Enforce uniform spacing at each level by shifting subtrees
        self._enforce_uniform_spacing(nodes_in_level, children_map, positions, spacing)

        max_cols = max((col + 1 for col, _ in positions.values()), default=1)
        return positions, max_cols

    def _enforce_uniform_spacing(
        self,
        nodes_in_level: dict[int, list[str]],
        children_map: dict[str, list[str]],
        positions: dict[str, tuple[int, int]],
        spacing: int,
    ) -> None:
        """Enforce uniform spacing between siblings at each level by shifting subtrees.

        After the initial layout, this method ensures that all nodes at each level
        maintain the required spacing from their left sibling. When a node is too close
        to its left sibling, the entire subtree rooted at that node is shifted right.

        Args:
            nodes_in_level: Mapping of level to list of node IDs at that level.
            children_map: Mapping of node ID to list of child IDs.
            positions: Dictionary of node positions to update in-place.
            spacing: Required spacing between siblings.
        """
        # Process each level from top to bottom
        for level in sorted(nodes_in_level.keys()):
            nodes = nodes_in_level[level]

            # Sort nodes by their current column position to process left-to-right
            sorted_nodes = sorted(nodes, key=lambda n: positions[n][0])

            # Track the minimum required column for the next node
            min_next_col = None

            for node_id in sorted_nodes:
                col, row = positions[node_id]

                if min_next_col is not None and col != min_next_col:
                    # This node needs to be at exactly min_next_col to maintain uniform spacing
                    shift_amount = min_next_col - col
                    self._shift_subtree(node_id, shift_amount, children_map, positions)
                    col = min_next_col

                # Update minimum column for next node at this level
                min_next_col = col + spacing

    def _shift_subtree(
        self,
        node_id: str,
        shift_amount: int,
        children_map: dict[str, list[str]],
        positions: dict[str, tuple[int, int]],
    ) -> None:
        """Shift a node and all its descendants horizontally by shift_amount.

        Args:
            node_id: Root of the subtree to shift.
            shift_amount: Amount to shift (positive = right, negative = left).
            children_map: Mapping of node ID to list of child IDs.
            positions: Dictionary containing node positions to update.
        """
        if node_id not in positions:
            return

        # Shift this node
        col, row = positions[node_id]
        positions[node_id] = (col + shift_amount, row)

        # Recursively shift all children
        for child in children_map.get(node_id, []):
            self._shift_subtree(child, shift_amount, children_map, positions)

    def _layout_subtree_uniform(
        self,
        node_id: str,
        level: int,
        children_map: dict[str, list[str]],
        positions: dict[str, tuple[int, int]],
        next_col: list[int],
        spacing: int,
    ) -> None:
        """Layout subtree using bottom-up traversal (Pass 1 of 2).

        Assigns initial x-positions during in-order traversal. Leaves get sequential
        positions with spacing between them, parents are centered over children.
        A second pass (_enforce_uniform_spacing) will adjust positions to ensure
        uniform spacing at all levels.

        Args:
            node_id: Root of the subtree to layout.
            level: Current level/row of this node.
            children_map: Mapping of node ID to list of child IDs.
            positions: Dictionary to populate with positions.
            next_col: List containing next available column (mutable reference).
            spacing: Uniform spacing between leaf siblings.
        """
        children = children_map.get(node_id, [])

        if not children:
            # Leaf node: assign next available column with spacing
            positions[node_id] = (next_col[0], level)
            next_col[0] += spacing
            return

        # Process all children first to get their positions
        child_cols: list[int] = []
        for child in children:
            self._layout_subtree_uniform(child, level + 1, children_map, positions, next_col, spacing)
            child_cols.append(positions[child][0])

        # Center parent over children (exact midpoint)
        parent_col = (min(child_cols) + max(child_cols)) // 2
        positions[node_id] = (parent_col, level)

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

        The connection goes straight down from parent, turns horizontally toward child,
        then goes straight up to child. This creates cleaner paths for compact layouts.

        Args:
            ax: Matplotlib axes object.
            x1: X-coordinate of first point (parent bottom center).
            y1: Y-coordinate of first point (parent bottom).
            x2: X-coordinate of second point (child top center).
            y2: Y-coordinate of second point (child top).

        """
        # Use class constants for connection styling
        curve_radius = TreeGrid.CONNECTION_CURVE_RADIUS
        line_width = TreeGrid.CONNECTION_LINE_WIDTH
        line_color = TreeGrid.CONNECTION_LINE_COLOR

        # Calculate the y-coordinates for the horizontal segment
        # Place it closer to the child to minimize horizontal extension
        vertical_gap = y1 - y2
        horizontal_y = y2 + vertical_gap * 0.3  # 30% down from parent toward child

        # Determine horizontal direction
        curve_sign = 1 if x2 > x1 else -1
        x_diff = abs(x2 - x1)

        # Create path with curved corners using Bezier curves
        verts = []
        codes = []

        # Start point (bottom of parent node)
        verts.append((x1, y1))
        codes.append(Path.MOVETO)

        # If nodes are aligned vertically (or very close), just draw a straight line
        if x_diff < curve_radius * 2:
            verts.append((x2, y2))
            codes.append(Path.LINETO)
        else:
            # Vertical line down to curve start
            verts.append((x1, horizontal_y + curve_radius))
            codes.append(Path.LINETO)

            # Curve from vertical to horizontal (first corner)
            TreeGrid._add_curve(verts, codes, x1, horizontal_y, curve_sign * curve_radius, 0)

            # Horizontal line
            verts.append((x2 - (curve_sign * curve_radius), horizontal_y))
            codes.append(Path.LINETO)

            # Curve from horizontal to vertical (second corner)
            TreeGrid._add_curve(verts, codes, x2, horizontal_y, 0, -curve_radius)

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

    def _get_responsive_font_sizes(self, styling_context: Any) -> dict[str, float]:  # noqa: ANN401
        """Calculate responsive font sizes based on node dimensions.

        Scales font sizes proportionally to node dimensions relative to defaults.
        Uses the smaller of width/height scaling to ensure text fits.

        Args:
            styling_context: Styling context containing default font sizes.

        Returns:
            dict[str, float]: Dictionary with 'label_size' and 'tick_size' keys.

        """
        # Default dimensions (what the fixed font sizes were designed for)
        default_width = 3.5
        default_height = 2.5

        # Calculate scaling factors
        width_scale = self.NODE_WIDTH / default_width
        height_scale = self.NODE_HEIGHT / default_height

        # Use the smaller scale factor to ensure text fits
        scale_factor = min(width_scale, height_scale)

        # Scale the default font sizes
        base_label_size = styling_context.fonts.label_size
        base_tick_size = styling_context.fonts.tick_size

        return {
            "label_size": base_label_size * scale_factor,
            "tick_size": base_tick_size * scale_factor,
        }

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

        # Calculate responsive font sizes based on node dimensions
        font_sizes = self._get_responsive_font_sizes(styling_context)

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
            fontsize=font_sizes["label_size"],
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
                fontsize=font_sizes["tick_size"],
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
                fontsize=font_sizes["label_size"],
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
                fontsize=font_sizes["tick_size"],
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
                fontsize=font_sizes["tick_size"],
                color=data_text_color,
            )
