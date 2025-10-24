"""Module for creating revenue tree diagrams using matplotlib."""

from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path

from pyretailscience.plots.styles.styling_context import get_styling_context


class CustomRoundedBox(mpatches.PathPatch):
    """Custom patch with independent corner rounding for top and bottom."""

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

        # Bottom left corner
        if bottom_radius > 0:
            theta = np.linspace(np.pi, 3 * np.pi / 2, 10)
            verts.extend(
                [
                    (x + bottom_radius + bottom_radius * np.cos(t), y + bottom_radius + bottom_radius * np.sin(t))
                    for t in theta
                ],
            )
            codes.extend([Path.MOVETO] + [Path.LINETO] * 9)
        else:
            verts.append((x, y))
            codes.append(Path.MOVETO)

        # Bottom right corner
        if bottom_radius > 0:
            theta = np.linspace(3 * np.pi / 2, 2 * np.pi, 10)
            verts.extend(
                [
                    (
                        x + width - bottom_radius + bottom_radius * np.cos(t),
                        y + bottom_radius + bottom_radius * np.sin(t),
                    )
                    for t in theta
                ],
            )
            codes.extend([Path.LINETO] * 10)
        else:
            verts.append((x + width, y))
            codes.append(Path.LINETO)

        # Top right corner
        if top_radius > 0:
            theta = np.linspace(0, np.pi / 2, 10)
            verts.extend(
                [
                    (x + width - top_radius + top_radius * np.cos(t), y + height - top_radius + top_radius * np.sin(t))
                    for t in theta
                ],
            )
            codes.extend([Path.LINETO] * 10)
        else:
            verts.append((x + width, y + height))
            codes.append(Path.LINETO)

        # Top left corner
        if top_radius > 0:
            theta = np.linspace(np.pi / 2, np.pi, 10)
            verts.extend(
                [
                    (x + top_radius + top_radius * np.cos(t), y + height - top_radius + top_radius * np.sin(t))
                    for t in theta
                ],
            )
            codes.extend([Path.LINETO] * 10)
        else:
            verts.append((x, y + height))
            codes.append(Path.LINETO)

        codes.append(Path.CLOSEPOLY)
        verts.append((0, 0))

        path = Path(verts, codes)
        super().__init__(path, **kwargs)


def get_color(percent_change: float) -> str:
    """Return color based on percent change thresholds.

    Green if > 1%, Red if < -1%, Grey if between -1% and 1%.

    Args:
        percent_change: Percentage change value.

    Returns:
        Hex color code as string.

    """
    if percent_change > 1:
        return "#28a745"  # Green
    if percent_change < -1:
        return "#dc3545"  # Red
    return "#95a5a6"  # Grey


def create_tree_node(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    percent: float,
    value1: str,
    value2: str,
    top_radius: float = 0.15,
    bottom_radius: float = 0.15,
    styling_context: Any | None = None,  # noqa: ANN401
) -> None:
    """Create a tree node with custom rounded corners and standard fonts.

    Args:
        ax: Matplotlib axes object.
        x: X-coordinate of bottom-left corner.
        y: Y-coordinate of bottom-left corner.
        width: Width of the node.
        height: Height of the node.
        title: Title text for the node header.
        percent: Percentage change value.
        value1: First value text.
        value2: Second value text.
        top_radius: Radius for top corners.
        bottom_radius: Radius for bottom corners.
        styling_context: Optional styling context object.

    """
    if styling_context is None:
        styling_context = get_styling_context()

    # Get standard font properties
    semi_bold_font = styling_context.get_font_properties(styling_context.fonts.title_font)  # poppins_semi_bold
    regular_font = styling_context.get_font_properties(styling_context.fonts.label_font)  # poppins_regular

    # Determine color based on percent change
    color = get_color(percent)

    # Header section
    header_height = height * 0.4
    header_box = CustomRoundedBox(
        (x, y + height - header_height),
        width,
        header_height,
        top_radius=top_radius,
        bottom_radius=0,
        facecolor="#1E3A8A",
        edgecolor="none",
        linewidth=0,
    )
    ax.add_patch(header_box)

    # Data section
    data_height = height - header_height
    data_box = CustomRoundedBox(
        (x, y),
        width,
        data_height,
        top_radius=0,
        bottom_radius=bottom_radius,
        facecolor=color,
        edgecolor="none",
        linewidth=0,
    )
    ax.add_patch(data_box)

    # Header text - USE LABEL SIZE instead of title size to fit properly
    ax.text(
        x + width / 2,
        y + height - header_height / 2,
        title,
        ha="center",
        va="center",
        fontproperties=semi_bold_font,
        fontsize=styling_context.fonts.label_size,  # Changed to label size to fit
        color="white",
    )

    # Percentage text - keep title size
    ax.text(
        x + width / 4,
        y + data_height / 2,
        f"{percent:+.1f}%",
        ha="center",
        va="center",
        fontproperties=semi_bold_font,
        fontsize=styling_context.fonts.title_size,
        color="white",
    )

    # Value text - label size
    ax.text(
        x + 3 * width / 4,
        y + data_height / 2 + 0.1,
        value1,
        ha="center",
        va="center",
        fontproperties=regular_font,
        fontsize=styling_context.fonts.label_size,
        color="white",
    )
    ax.text(
        x + 3 * width / 4,
        y + data_height / 2 - 0.1,
        value2,
        ha="center",
        va="center",
        fontproperties=regular_font,
        fontsize=styling_context.fonts.label_size,
        color="white",
    )


def draw_connection(ax: Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    """Draw connection line between nodes.

    Args:
        ax: Matplotlib axes object.
        x1: X-coordinate of first point.
        y1: Y-coordinate of first point.
        x2: X-coordinate of second point.
        y2: Y-coordinate of second point.

    """
    mid_y = (y1 + y2) / 2
    ax.plot([x1, x1], [y1, mid_y], "k-", linewidth=2)
    ax.plot([x1, x2], [mid_y, mid_y], "k-", linewidth=2)
    ax.plot([x2, x2], [mid_y, y2], "k-", linewidth=2)


def create_revenue_tree() -> tuple[Figure, Axes]:
    """Create the revenue tree diagram.

    Returns:
        Tuple of (figure, axes) objects.

    """
    # Get styling context for standard fonts
    styling_context = get_styling_context()
    title_font = styling_context.get_font_properties(styling_context.fonts.title_font)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title (using standard title font and size)
    ax.text(
        7,
        9.5,
        "Matplotlib Tree Diagram - Option 4 Style",
        ha="center",
        va="center",
        fontproperties=title_font,
        fontsize=styling_context.fonts.title_size,
    )

    # Create nodes with proper sizing
    create_tree_node(
        ax,
        5.5,
        7.5,
        3,
        1.2,
        "Total Sales (TISP)",
        6.0,
        "£61.5m",
        "£58.1m",
        styling_context=styling_context,
    )

    create_tree_node(
        ax,
        1.5,
        5.5,
        3,
        1.2,
        "Non-Card Sales (TISP)",
        -0.9,
        "£24.1m",
        "£24.3m",
        styling_context=styling_context,
    )

    create_tree_node(
        ax,
        9.5,
        5.5,
        3,
        1.2,
        "Ad Card Sales (TISP)",
        11.0,
        "£37.4m",
        "£33.7m",
        styling_context=styling_context,
    )

    create_tree_node(ax, 1.5, 3.5, 3, 1.2, "Region: West", -15.3, "£12.5m", "£14.8m", styling_context=styling_context)

    create_tree_node(
        ax,
        7.5,
        3.5,
        3,
        1.2,
        "Product Category A",
        18.5,
        "£22.1m",
        "£18.6m",
        styling_context=styling_context,
    )

    create_tree_node(
        ax,
        11.5,
        3.5,
        2.5,
        1.2,
        "Product Category B",
        5.2,
        "£15.3m",
        "£14.5m",
        styling_context=styling_context,
    )

    create_tree_node(
        ax,
        6.0,
        1.5,
        2.5,
        1.2,
        "Sub-Category 1",
        25.0,
        "£13.2m",
        "£10.6m",
        styling_context=styling_context,
    )

    create_tree_node(ax, 9.0, 1.5, 2.5, 1.2, "Sub-Category 2", 12.8, "£8.9m", "£7.9m", styling_context=styling_context)

    # Draw connections
    draw_connection(ax, 7, 7.5, 3, 6.7)
    draw_connection(ax, 7, 7.5, 11, 6.7)
    draw_connection(ax, 3, 5.5, 3, 4.7)
    draw_connection(ax, 11, 5.5, 9, 4.7)
    draw_connection(ax, 11, 5.5, 12.75, 4.7)
    draw_connection(ax, 9, 3.5, 7.25, 2.7)
    draw_connection(ax, 9, 3.5, 10.25, 2.7)

    plt.tight_layout()
    plt.savefig("tree_diagram_matplotlib.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig, ax


if __name__ == "__main__":
    create_revenue_tree()
