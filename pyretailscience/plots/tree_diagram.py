"""Tree Diagram Module.

This module implements tree visualization components using matplotlib for creating
hierarchical tree diagrams with custom node types and grid-based layouts.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
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
        width: float | None = None,
        height: float | None = None,
    ) -> None:
        """Initialize the tree node.

        Args:
            data: Dictionary containing node data. Each subclass defines required keys.
            x: X-coordinate of bottom-left corner.
            y: Y-coordinate of bottom-left corner.
            width: Optional override for node width. If None, uses class default NODE_WIDTH.
            height: Optional override for node height. If None, uses class default NODE_HEIGHT.

        """
        self._data = data
        self.x = x
        self.y = y
        # Use instance-specific dimensions if provided, otherwise fall back to class constants
        # This allows TreeGrid to override dimensions per-node without mutating class attributes
        # Set instance dimensions (use class constants as fallback)
        # All render code should use self.node_width/node_height instead of NODE_WIDTH/NODE_HEIGHT
        self.node_width = width if width is not None else self.NODE_WIDTH
        self.node_height = height if height is not None else self.NODE_HEIGHT

        # For backward compatibility, keep NODE_WIDTH/NODE_HEIGHT accessible as instance attributes
        # This allows existing render code to work without changes
        self.NODE_WIDTH = self.node_width
        self.NODE_HEIGHT = self.node_height

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

    # Default dimensions used for responsive font scaling
    DEFAULT_WIDTH = 3.75
    DEFAULT_HEIGHT = 1.75

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
        # Calculate scaling factors relative to default dimensions
        width_scale = self.NODE_WIDTH / self.DEFAULT_WIDTH
        height_scale = self.NODE_HEIGHT / self.DEFAULT_HEIGHT

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
            ax: Matplotlib axes object to render on.
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
        """Return the node width."""
        return self.NODE_WIDTH

    def get_height(self) -> float:
        """Return the node height."""
        return self.NODE_HEIGHT


class SegmentTreeNode(TreeNode):
    """Segment analysis tree node for descriptive analytics and cohort comparison.

    Designed for showing segment metrics compared to a baseline (e.g., survival rates,
    conversion rates, or other KPIs for different customer segments).

    Required data keys:
        header: str - Segment name (e.g., "Female Passengers", "Premium Customers")
        metric_label: str - Name of the metric being analyzed (e.g., "Survival Rate", "Conversion Rate")
        metric_value: str - The metric value for this segment (e.g., "74.2%", "£125.50")
        sample_size: str - Number of items in segment (e.g., "271", "1,234 customers")
        baseline_value: str - The baseline/overall value for comparison (e.g., "40.6%")
        variance: str - Difference from baseline (e.g., "+33.6pp", "-15.2%")

    Optional data keys:
        baseline_label: str - Label for baseline (default: "Overall")
        contribution: str - Contribution to total (e.g., "69.3% of survivors")
        contribution_label: str - Label for contribution (default: "of total")
        variance_numeric: float - Numeric variance for color coding (extracted from variance if not provided)

    Example:
        >>> tree_structure = {
        ...     "all": {
        ...         "header": "All Customers",
        ...         "metric_label": "Conversion Rate",
        ...         "metric_value": "12.5%",
        ...         "sample_size": "10,000",
        ...         "baseline_value": "12.5%",
        ...         "variance": "0.0pp",
        ...         "contribution": "100%",
        ...         "children": ["premium", "standard"]
        ...     },
        ...     "premium": {
        ...         "header": "Premium",
        ...         "metric_label": "Conversion Rate",
        ...         "metric_value": "28.3%",
        ...         "sample_size": "2,500",
        ...         "baseline_value": "12.5%",
        ...         "variance": "+15.8pp",
        ...         "contribution": "56.7%",
        ...         "children": []
        ...     }
        ... }
        >>> grid = TreeGrid(tree_structure, SegmentTreeNode)
        >>> grid.render()
    """

    NODE_WIDTH = 3.5
    NODE_HEIGHT = 2.2

    # Default dimensions used for responsive font scaling
    DEFAULT_WIDTH = 3.5
    DEFAULT_HEIGHT = 2.2

    # Color thresholds for variance (percentage points or percentage)
    GREEN_THRESHOLD = 5.0  # Variance at or above this shows green
    RED_THRESHOLD = -5.0  # Variance at or below this shows red

    @staticmethod
    def _get_color(variance: float) -> str:
        """Return color based on variance from baseline.

        Green if >= GREEN_THRESHOLD, Red if <= RED_THRESHOLD, Yellow otherwise.

        Args:
            variance: Numeric variance from baseline.

        Returns:
            str: Hex color code as string.
        """
        if variance >= SegmentTreeNode.GREEN_THRESHOLD:
            return COLORS["green"][500]
        if variance <= SegmentTreeNode.RED_THRESHOLD:
            return COLORS["red"][500]
        return COLORS["yellow"][500]

    def _get_responsive_font_sizes(self) -> dict[str, float]:
        """Calculate responsive font sizes based on node dimensions.

        Scales fonts relative to the default dimensions to maintain readability
        when node sizes are customized.

        Returns:
            dict[str, float]: Font sizes for different text elements.
        """
        width_scale = self.NODE_WIDTH / self.DEFAULT_WIDTH
        height_scale = self.NODE_HEIGHT / self.DEFAULT_HEIGHT
        scale_factor = min(width_scale, height_scale)

        return {
            "header_size": 11 * scale_factor,
            "metric_size": 10 * scale_factor,
            "label_size": 8.5 * scale_factor,
        }

    def render(self, ax: Axes) -> None:
        """Render the segment analysis node on the given axes.

        Args:
            ax: Matplotlib axes object to render on.
        """
        data = self._data

        # Extract required fields
        header = data["header"]
        metric_label = data["metric_label"]
        metric_value = data["metric_value"]
        sample_size = data["sample_size"]
        baseline_value = data["baseline_value"]
        variance = data["variance"]

        # Extract optional fields
        baseline_label = data.get("baseline_label", "Overall")
        contribution = data.get("contribution")
        contribution_label = data.get("contribution_label", "of total")

        # Extract numeric variance for color coding
        if "variance_numeric" in data:
            variance_numeric = data["variance_numeric"]
        else:
            # Try to extract numeric value from variance string
            import re

            match = re.search(r"([+-]?\d+\.?\d*)", variance)
            variance_numeric = float(match.group(1)) if match else 0.0

        # Layout constants
        corner_radius = 0.15
        header_height_ratio = 0.28
        header_height = self.NODE_HEIGHT * header_height_ratio
        content_height = self.NODE_HEIGHT - header_height

        # Colors
        header_color = self._get_color(variance_numeric)
        header_text_color = "white"
        content_bg_color = "white"
        text_color = COLORS["gray"][700]
        border_color = COLORS["gray"][700]

        # Get responsive font sizes
        font_sizes = self._get_responsive_font_sizes()

        # Draw header box with colored background
        header_box = BaseRoundedBox(
            xy=(self.x, self.y + content_height),
            width=self.NODE_WIDTH,
            height=header_height,
            top_radius=corner_radius,
            bottom_radius=0,
            facecolor=header_color,
            edgecolor=border_color,
            linewidth=1.5,
        )
        ax.add_patch(header_box)

        # Draw content box
        content_box = BaseRoundedBox(
            xy=(self.x, self.y),
            width=self.NODE_WIDTH,
            height=content_height,
            top_radius=0,
            bottom_radius=corner_radius,
            facecolor=content_bg_color,
            edgecolor=border_color,
            linewidth=1.5,
        )
        ax.add_patch(content_box)

        # Render header text
        ax.text(
            self.x + self.NODE_WIDTH / 2,
            self.y + content_height + header_height / 2,
            header,
            ha="center",
            va="center",
            fontsize=font_sizes["header_size"],
            fontweight="bold",
            color=header_text_color,
        )

        # Render content area text
        y_offset = self.y + content_height * 0.88
        line_height = content_height * 0.20

        # Line 1: Metric label and value
        ax.text(
            self.x + self.NODE_WIDTH * 0.05,
            y_offset,
            f"{metric_label}: {metric_value}",
            ha="left",
            va="center",
            fontsize=font_sizes["metric_size"],
            fontweight="bold",
            color=text_color,
        )
        y_offset -= line_height

        # Line 2: Sample size
        ax.text(
            self.x + self.NODE_WIDTH * 0.05,
            y_offset,
            f"Sample: {sample_size}",
            ha="left",
            va="center",
            fontsize=font_sizes["label_size"],
            color=text_color,
        )
        y_offset -= line_height

        # Line 3: Variance from baseline
        ax.text(
            self.x + self.NODE_WIDTH * 0.05,
            y_offset,
            f"vs {baseline_label} ({baseline_value}): {variance}",
            ha="left",
            va="center",
            fontsize=font_sizes["label_size"],
            color=text_color,
        )
        y_offset -= line_height

        # Line 4: Contribution (if provided)
        if contribution is not None:
            ax.text(
                self.x + self.NODE_WIDTH * 0.05,
                y_offset,
                f"{contribution} {contribution_label}",
                ha="left",
                va="center",
                fontsize=font_sizes["label_size"],
                color=text_color,
            )

    def get_width(self) -> float:
        """Return the node width."""
        return self.NODE_WIDTH

    def get_height(self) -> float:
        """Return the node height."""
        return self.NODE_HEIGHT


class TreeGrid:
    """Grid-based tree diagram renderer with configurable node types."""

    # Connection styling constants
    CONNECTION_CURVE_RADIUS = 0.15
    CONNECTION_LINE_WIDTH = 2
    CONNECTION_LINE_COLOR = "black"

    # Layout constants
    POSITION_TOLERANCE = 0.01  # Minimum change threshold for parent centering convergence

    def __init__(
        self,
        tree_structure: dict[str, dict],
        node_class: type[TreeNode],
        node_width: float | None = None,
        node_height: float | None = None,
        vertical_spacing: float | None = None,
        horizontal_spacing: float | None = None,
        grid_spacing: int = 2,
        orientation: str = "top-down",
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
            grid_spacing: Grid cell spacing between sibling nodes. Default is 2 (one empty cell between nodes).
                Use 1 for compact layouts (nodes directly adjacent with minimal gap).
            orientation: Tree orientation. 'top-down' (default) for vertical trees, 'left-right' for horizontal trees.

        Raises:
            ValueError: If tree_structure is empty, spacing is insufficient, or orientation is invalid.
            TypeError: If node_class is not a TreeNode subclass.

        """
        # Validate node_class is a TreeNode subclass
        if not issubclass(node_class, TreeNode):
            error_msg = f"node_class must be a TreeNode subclass, got {node_class}"
            raise TypeError(error_msg)

        # Validate tree_structure is not empty
        if not tree_structure:
            raise ValueError("tree_structure cannot be empty")

        # Validate orientation
        valid_orientations = {"top-down", "left-right"}
        if orientation not in valid_orientations:
            error_msg = f"orientation must be one of {valid_orientations}, got '{orientation}'"
            raise ValueError(error_msg)

        self.tree_structure = tree_structure
        self.node_class = node_class
        self.orientation = orientation

        # Get node dimensions from the node class or use overrides
        self.node_width = node_width if node_width is not None else node_class.NODE_WIDTH
        self.node_height = node_height if node_height is not None else node_class.NODE_HEIGHT

        # Auto-calculate spacing if not provided
        # Default: 10% of node width horizontally, slightly more vertically
        self.vertical_spacing = vertical_spacing if vertical_spacing is not None else self.node_height + 0.8
        self.horizontal_spacing = horizontal_spacing if horizontal_spacing is not None else self.node_width * 1.1
        self.grid_spacing = grid_spacing

        # Validate spacing is sufficient to prevent node overlap
        if self.horizontal_spacing < self.node_width:
            error_msg = (
                f"horizontal_spacing ({self.horizontal_spacing}) must be >= "
                f"node_width ({self.node_width}) to prevent node overlap"
            )
            raise ValueError(error_msg)
        if self.vertical_spacing < self.node_height:
            error_msg = (
                f"vertical_spacing ({self.vertical_spacing}) must be >= "
                f"node_height ({self.node_height}) to prevent node overlap"
            )
            raise ValueError(error_msg)

        # Always compute positions automatically from tree structure
        positions, computed_rows, computed_cols = self._compute_positions()

        # Store computed positions (we'll use them during rendering)
        self._positions = positions
        self.num_rows = computed_rows
        self.num_cols = computed_cols

        # Generate row and column positions based on orientation
        if self.orientation == "top-down":
            # Row 0 is at the top, increasing downward (reversed from matplotlib's default bottom-up)
            self.row = {i: (self.num_rows - 1 - i) * self.vertical_spacing for i in range(self.num_rows)}
            self.col = {i: i * self.horizontal_spacing for i in range(self.num_cols)}
        else:  # left-right
            # Row 0 is on the left, increasing rightward (row = depth, col = vertical position)
            # Use horizontal_spacing for depth (x-axis) and vertical_spacing for siblings (y-axis)
            self.row = {i: i * self.horizontal_spacing for i in range(self.num_rows)}
            self.col = {i: (self.num_cols - 1 - i) * self.vertical_spacing for i in range(self.num_cols)}

    def render(self, ax: Axes | None = None) -> Axes:  # noqa: C901
        """Render the tree diagram.

        Args:
            ax: Optional matplotlib axes object. If None, creates a new figure and axes.

        Returns:
            The matplotlib axes object.

        """
        if ax is None:
            # Calculate plot dimensions based on orientation
            if self.orientation == "top-down":
                # Find the maximum x position from all nodes
                max_x = max(col_idx * self.horizontal_spacing for col_idx, _ in self._positions.values())
                plot_width = max_x + self.node_width
                # Row 0 is at the top with the highest y-value after coordinate inversion
                plot_height = self.row[0] + self.node_height
            else:  # left-right
                # Find the maximum y position (which comes from col) - siblings use vertical_spacing
                max_y = max(col_idx * self.vertical_spacing for col_idx, _ in self._positions.values())
                plot_height = max_y + self.node_height
                # Find the maximum x position (rightmost row) - depth uses horizontal_spacing
                max_x = max(self.row[row_idx] for _, row_idx in self._positions.values())
                plot_width = max_x + self.node_width

            _, ax = plt.subplots(figsize=(plot_width, plot_height))
            ax.set_xlim(0, plot_width)
            ax.set_ylim(0, plot_height)
            ax.axis("off")

        # First pass: create all nodes and store their centers
        node_centers = {}
        for node_id, node_data in self.tree_structure.items():
            # Get computed position from auto-layout
            col_idx, row_idx = self._positions[node_id]

            # Calculate x and y based on orientation
            if self.orientation == "top-down":
                x = col_idx * self.horizontal_spacing  # Calculate x from float column index
                y = self.row[row_idx]
            else:  # left-right
                x = self.row[row_idx]  # Depth (row) uses horizontal_spacing
                y = col_idx * self.vertical_spacing  # Siblings (col) use vertical_spacing

            # Extract data for the node (exclude children which is structural)
            data_dict = {k: v for k, v in node_data.items() if k != "children"}

            # Create and render the node with custom dimensions if specified
            node = self.node_class(
                data=data_dict,
                x=x,
                y=y,
                width=self.node_width,
                height=self.node_height,
            )
            node.render(ax)

            # Store center positions for connections based on orientation
            if self.orientation == "top-down":
                node_centers[node_id] = {
                    "x": x + self.node_width / 2,
                    "y_bottom": y,
                    "y_top": y + self.node_height,
                }
            else:  # left-right
                node_centers[node_id] = {
                    "y": y + self.node_height / 2,
                    "x_left": x,
                    "x_right": x + self.node_width,
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

                    if self.orientation == "top-down":
                        self._draw_connection(
                            ax=ax,
                            x1=parent["x"],
                            y1=parent["y_bottom"],
                            x2=child["x"],
                            y2=child["y_top"],
                            orientation="top-down",
                        )
                    else:  # left-right
                        self._draw_connection(
                            ax=ax,
                            x1=parent["x_right"],
                            y1=parent["y"],
                            x2=child["x_left"],
                            y2=child["y"],
                            orientation="left-right",
                        )

        return ax

    def _compute_positions(self) -> tuple[dict[str, tuple[float, int]], int, int]:
        """Compute grid positions using Excel-like grid layout system.

        Implements a grid-based layout where:
        - Each node occupies a (row, col) coordinate in a virtual grid
        - Parents are centered at the exact average column of their children
        - Nodes in the same row maintain consistent spacing (at most 1 cell apart)
        - Bottom-up processing ensures children are positioned before parents
        - Float precision maintained for accurate parent centering

        Layout orientation: top-down (root at row 0), left-to-right (siblings ordered left to right).

        Returns:
            positions: mapping of node_id -> (col, row) where col is float for precise centering
            num_rows: total number of rows required
            num_cols: maximum number of columns across rows
        """
        # Build parent-child relationships
        children_map: dict[str, list[str]] = {}
        referenced_as_child: set[str] = set()
        for node_id, node_data in self.tree_structure.items():
            children = node_data.get("children", []) or []
            children_map[node_id] = list(children)
            for c in children:
                referenced_as_child.add(c)

        # Find root nodes (never referenced as children)
        roots = [nid for nid in self.tree_structure if nid not in referenced_as_child]
        if not roots:
            # Fallback: use first node as root
            roots = [sorted(self.tree_structure.keys())[0]]

        # Assign rows using level-order traversal
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

        # Group nodes by level (row) with stable ordering
        nodes_in_level: dict[int, list[str]] = defaultdict(list)
        for nid in self.tree_structure:
            lvl = level_of.get(nid, 0)
            nodes_in_level[lvl].append(nid)

        # Compute grid positions using Excel-like grid system
        positions = self._compute_grid_positions(nodes_in_level, children_map)

        # Calculate grid dimensions
        num_rows = len(nodes_in_level.keys()) if nodes_in_level else 1

        # Find maximum column index (accounting for float precision)
        max_col = 0.0
        for col, _row in positions.values():
            max_col = max(max_col, col)
        # Convert to integer column count (round up to nearest integer)
        num_cols = int(max_col) + 1 if positions else 1

        return positions, num_rows, num_cols

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
    def _draw_connection(ax: Axes, x1: float, y1: float, x2: float, y2: float, orientation: str = "top-down") -> None:
        """Draw connection line between nodes with curved corners.

        For top-down: goes straight down from parent, turns horizontally toward child, then up to child.
        For left-right: goes straight right from parent, turns vertically toward child, then right to child.

        Args:
            ax: Matplotlib axes object.
            x1: X-coordinate of first point (parent exit point).
            y1: Y-coordinate of first point (parent exit point).
            x2: X-coordinate of second point (child entry point).
            y2: Y-coordinate of second point (child entry point).
            orientation: Tree orientation ('top-down' or 'left-right').

        """
        # Use class constants for connection styling
        curve_radius = TreeGrid.CONNECTION_CURVE_RADIUS
        line_width = TreeGrid.CONNECTION_LINE_WIDTH
        line_color = TreeGrid.CONNECTION_LINE_COLOR

        # Create path with curved corners using Bezier curves
        verts = []
        codes = []

        # Start point
        verts.append((x1, y1))
        codes.append(Path.MOVETO)

        if orientation == "top-down":
            # Calculate the y-coordinate for the horizontal segment
            vertical_gap = y1 - y2
            horizontal_y = y2 + vertical_gap * 0.3  # 30% down from parent toward child

            # Determine horizontal direction
            curve_sign = 1 if x2 > x1 else -1
            x_diff = abs(x2 - x1)

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

        else:  # left-right
            # Calculate the x-coordinate for the vertical segment
            horizontal_gap = x2 - x1
            vertical_x = x1 + horizontal_gap * 0.3  # 30% right from parent toward child

            # Determine vertical direction
            curve_sign = 1 if y2 > y1 else -1
            y_diff = abs(y2 - y1)

            # If nodes are aligned horizontally (or very close), just draw a straight line
            if y_diff < curve_radius * 2:
                verts.append((x2, y2))
                codes.append(Path.LINETO)
            else:
                # Small gap to ensure line starts outside parent box
                gap = 0.05
                # Horizontal line right to curve start
                verts.append((max(x1 + gap, vertical_x - curve_radius), y1))
                codes.append(Path.LINETO)

                # Curve from horizontal to vertical (first corner)
                TreeGrid._add_curve(verts, codes, vertical_x, y1, 0, curve_sign * curve_radius)

                # Vertical line
                verts.append((vertical_x, y2 - (curve_sign * curve_radius)))
                codes.append(Path.LINETO)

                # Curve from vertical to horizontal (second corner)
                TreeGrid._add_curve(verts, codes, vertical_x, y2, curve_radius, 0)

                # Horizontal line right to child node
                verts.append((x2, y2))
                codes.append(Path.LINETO)

        # Create and draw the path
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="none", edgecolor=line_color, linewidth=line_width)
        ax.add_patch(patch)

    def _compute_grid_positions(  # noqa: C901
        self,
        nodes_in_level: dict[int, list[str]],
        children_map: dict[str, list[str]],
    ) -> dict[str, tuple[float, int]]:
        """Compute positions using Excel-like grid system with tree-based layout.

        Implements a clean grid-based layout algorithm:
        1. Tree traversal: process nodes in tree order (depth-first) to assign initial columns
        2. Leaf nodes: placed at consecutive columns with spacing of 2 (at most 1 empty cell between)
        3. Internal nodes: centered at exact average column of their children
        4. Spacing enforcement: maintains "at most 1 cell apart" within each row
        5. Float precision: maintains accurate centering for parent nodes

        Args:
            nodes_in_level: Mapping of level (row) to list of node IDs at that level.
            children_map: Mapping of node ID to list of child node IDs.

        Returns:
            Dictionary mapping node IDs to (col, row) grid positions with float column indices.
        """
        grid_positions: dict[str, tuple[float, int]] = {}
        spacing = self.grid_spacing  # Spacing between sibling nodes
        next_col = [0.0]  # Mutable reference for tracking next available leaf column

        # Find root nodes (nodes never referenced as children)
        all_children = {child for children in children_map.values() for child in children}
        roots = [
            node for level_nodes in [nodes_in_level.get(0, [])] for node in level_nodes if node not in all_children
        ]
        if not roots:
            roots = list(nodes_in_level.get(0, []))

        def layout_subtree(node_id: str, level: int) -> None:
            """Recursively layout subtree using depth-first traversal.

            Args:
                node_id: Current node to layout.
                level: Current level/row of this node.
            """
            children = children_map.get(node_id, [])

            if len(children) == 0:
                # Leaf node: assign next available column
                grid_positions[node_id] = (next_col[0], level)
                next_col[0] += spacing
            else:
                # Internal node: process children first, then center over them
                for child_id in children:
                    layout_subtree(child_id, level + 1)

                # Center parent at exact average of children columns
                child_cols = [grid_positions[child_id][0] for child_id in children]
                avg_col = sum(child_cols) / len(child_cols)
                grid_positions[node_id] = (avg_col, level)

        # Layout each root subtree
        for root in roots:
            layout_subtree(root, 0)

        # Iterative spacing enforcement: converge to valid layout
        # Keep adjusting until no more overlaps exist
        max_iterations = 10
        for _ in range(max_iterations):
            changes_made = False

            # Process from deepest level to shallowest (bottom-up)
            for level in sorted(nodes_in_level.keys(), reverse=True):
                nodes = nodes_in_level[level]
                sorted_nodes = sorted(nodes, key=lambda n: grid_positions[n][0])

                # Enforce minimum spacing to prevent overlaps
                for i in range(len(sorted_nodes) - 1):
                    curr_node = sorted_nodes[i]
                    next_node = sorted_nodes[i + 1]

                    curr_col = grid_positions[curr_node][0]
                    next_col_pos = grid_positions[next_node][0]

                    gap = next_col_pos - curr_col
                    min_gap = spacing  # Minimum spacing to prevent overlap

                    # If gap is too small, shift next node and its subtree right
                    if gap < min_gap - 0.01:  # Allow small floating point tolerance
                        shift_amount = min_gap - gap
                        self._shift_node_and_descendants(next_node, shift_amount, children_map, grid_positions)
                        changes_made = True

                # After adjusting spacing at this level, re-center parents at the level above
                if level > 0:
                    parent_level_nodes = nodes_in_level.get(level - 1, [])
                    for parent_id in parent_level_nodes:
                        children = children_map.get(parent_id, [])
                        if len(children) > 0:
                            child_cols = [grid_positions[child][0] for child in children if child in grid_positions]
                            if len(child_cols) > 0:
                                old_col = grid_positions[parent_id][0]
                                avg_col = sum(child_cols) / len(child_cols)
                                if abs(avg_col - old_col) > self.POSITION_TOLERANCE:
                                    _, parent_row = grid_positions[parent_id]
                                    grid_positions[parent_id] = (avg_col, parent_row)
                                    changes_made = True

            # If no changes were made in this iteration, we've converged
            if not changes_made:
                break

        return grid_positions

    def _shift_node_and_descendants(
        self,
        node_id: str,
        shift_amount: float,
        children_map: dict[str, list[str]],
        positions: dict[str, tuple[float, int]],
    ) -> None:
        """Recursively shift a node and all its descendants horizontally.

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

        # Recursively shift all descendants
        children = children_map.get(node_id, [])
        for child_id in children:
            self._shift_node_and_descendants(child_id, shift_amount, children_map, positions)


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

    # Default dimensions used for responsive font scaling
    DEFAULT_WIDTH = 3.5
    DEFAULT_HEIGHT = 2.5

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
        # Calculate scaling factors relative to default dimensions
        width_scale = self.NODE_WIDTH / self.DEFAULT_WIDTH
        height_scale = self.NODE_HEIGHT / self.DEFAULT_HEIGHT

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


def lightgbm_tree_to_grid(booster: Any, feature_names: list[str] | None = None) -> dict[str, dict]:  # noqa: C901, ANN401
    """Convert LightGBM single-tree booster to TreeGrid format.

    Parses the LightGBM tree structure and creates a dictionary compatible with TreeGrid.
    Automatically calculates value ranges for color normalization.

    Args:
        booster: LightGBM Booster object (must have single tree, n_estimators=1).
        feature_names: Optional list of feature names for display. If None, uses split indices.

    Returns:
        dict: Tree structure dict mapping node_id -> node data with children list.
              Each node contains: split_feature, split_condition, value, sample_count,
              avg_value, value_range, and children list.

    Raises:
        ValueError: If booster doesn't have exactly one tree.

    Example:
        >>> from lightgbm import LGBMRegressor
        >>> model = LGBMRegressor(n_estimators=1, max_depth=3)
        >>> model.fit(X_train, y_train)
        >>> tree_structure = lightgbm_tree_to_grid(model.booster_, feature_names=X_train.columns)
        >>> grid = TreeGrid(tree_structure=tree_structure, node_class=LightGBMTreeNode)
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

    # Convert tree structure to TreeGrid format
    tree_structure: dict[str, dict] = {}
    node_counter = [0]  # Use list for mutable counter

    def convert_node(lgbm_node: dict[str, Any]) -> str:
        """Recursively convert LightGBM node to TreeGrid format."""
        node_id = f"node_{node_counter[0]}"
        node_counter[0] += 1

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

            # Process children
            if "left_child" in lgbm_node:
                left_child_id = convert_node(lgbm_node["left_child"])
                node_data["children"].append(left_child_id)

            if "right_child" in lgbm_node:
                right_child_id = convert_node(lgbm_node["right_child"])
                node_data["children"].append(right_child_id)

        tree_structure[node_id] = node_data
        return node_id

    # Convert starting from root
    convert_node(tree_structure_raw)

    return tree_structure
