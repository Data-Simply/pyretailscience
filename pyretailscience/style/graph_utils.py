"""Helper functions for styling graphs."""

import importlib.resources as pkg_resources
from collections.abc import Generator
from itertools import cycle

import matplotlib.font_manager as fm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text

ASSETS_PATH = pkg_resources.files("pyretailscience").joinpath("assets")


def _hatches_gen() -> Generator[str, None, None]:
    """Returns a generator that cycles through predefined hatch patterns.

    Yields:
        str: The next hatch pattern in the sequence.
    """
    _hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    return cycle(_hatches)


class GraphStyles:
    """A class to hold the styles for a graph."""

    POPPINS_BOLD = fm.FontProperties(fname=f"{ASSETS_PATH}/fonts/Poppins-Bold.ttf")
    POPPINS_SEMI_BOLD = fm.FontProperties(fname=f"{ASSETS_PATH}/fonts/Poppins-SemiBold.ttf")
    POPPINS_REG = fm.FontProperties(fname=f"{ASSETS_PATH}/fonts/Poppins-Regular.ttf")
    POPPINS_MED = fm.FontProperties(fname=f"{ASSETS_PATH}/fonts/Poppins-Medium.ttf")
    POPPINS_LIGHT_ITALIC = fm.FontProperties(fname=f"{ASSETS_PATH}/fonts/Poppins-LightItalic.ttf")

    DEFAULT_TITLE_FONT_SIZE = 20
    DEFAULT_SOURCE_FONT_SIZE = 10
    DEFAULT_AXIS_LABEL_FONT_SIZE = 12
    DEFAULT_TICK_LABEL_FONT_SIZE = 10
    DEFAULT_BAR_LABEL_FONT_SIZE = 11

    DEFAULT_AXIS_LABEL_PAD = 10
    DEFAULT_TITLE_PAD = 10

    DEFAULT_BAR_WIDTH = 0.8


def human_format(
    num: float,
    pos: int | None = None,  # noqa: ARG001 (pos is only used for Matplotlib compatibility)
    decimals: int = 0,
    prefix: str = "",
) -> str:
    """Format a number in a human readable format for Matplotlib.

    Args:
        num (float): The number to format.
        pos (int, optional): The position. Defaults to None. Only used for Matplotlib compatibility.
        decimals (int, optional): The number of decimals. Defaults to 0.
        prefix (str, optional): The prefix of the returned string, eg '$'. Defaults to "".

    Returns:
        str: The formatted number.
    """
    # The minimum difference between two numbers to recieve a different suffix
    minimum_magnitude_difference = 1000.0

    magnitude = 0
    while abs(num) >= minimum_magnitude_difference:
        magnitude += 1
        num /= minimum_magnitude_difference

    # Add more suffixes if you need them
    return f"{prefix}%.{decimals}f%s" % (num, ["", "K", "M", "B", "T", "P"][magnitude])


def _add_plot_legend(ax: Axes, legend_title: str | None, move_legend_outside: bool) -> Axes:
    """Add a legend to a Matplotlib graph.

    Args:
        ax (Axes): The axes object of the plot.
        legend_title (str, optional): The title for the legend.
        move_legend_outside (bool, optional): Whether to move legend outside the plot.
    """
    has_legend = ax.get_legend() is not None
    if has_legend or legend_title is not None or move_legend_outside:
        legend = (
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
            if move_legend_outside
            else ax.legend(frameon=False)
        )
        if legend_title:
            legend.set_title(legend_title)
    return ax


def standard_graph_styles(
    ax: Axes,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title_pad: int = GraphStyles.DEFAULT_TITLE_PAD,
    x_label_pad: int = GraphStyles.DEFAULT_AXIS_LABEL_PAD,
    y_label_pad: int = GraphStyles.DEFAULT_AXIS_LABEL_PAD,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
) -> Axes:
    """Apply standard styles to a Matplotlib graph.

    Args:
        ax (Axes): The graph to apply the styles to.
        title (str, optional): The title of the graph. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None.
        y_label (str, optional): The y-axis label. Defaults to None.
        title_pad (int, optional): The padding above the title. Defaults to GraphStyles.DEFAULT_TITLE_PAD.
        x_label_pad (int, optional): The padding below the x-axis label. Defaults to GraphStyles.DEFAULT_AXIS_LABEL_PAD.
        y_label_pad (int, optional): The padding to the left of the y-axis label. Defaults to
            GraphStyles.DEFAULT_AXIS_LABEL_PAD.
        legend_title (str, optional): The title of the legend. If None, no legend title is applied. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot. Defaults to False.

    Returns:
        Axes: The graph with the styles applied.
    """
    ax.set_facecolor("w")  # set background color to white
    ax.set_axisbelow(True)  # set grid lines behind the plot
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", axis="x", color="#DAD8D7", alpha=0.5, zorder=1)
    ax.grid(which="major", axis="y", color="#DAD8D7", alpha=0.5, zorder=1)

    if title is not None:
        ax.set_title(
            title,
            fontproperties=GraphStyles.POPPINS_SEMI_BOLD,
            fontsize=GraphStyles.DEFAULT_TITLE_FONT_SIZE,
            pad=title_pad,
        )

    if x_label is not None:
        ax.set_xlabel(
            x_label,
            fontproperties=GraphStyles.POPPINS_REG,
            fontsize=GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE,
            labelpad=x_label_pad,
        )

    if y_label is not None:
        ax.set_ylabel(
            y_label,
            fontproperties=GraphStyles.POPPINS_REG,
            fontsize=GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE,
            labelpad=y_label_pad,
        )

    return _add_plot_legend(ax=ax, legend_title=legend_title, move_legend_outside=move_legend_outside)


def standard_tick_styles(ax: Axes) -> Axes:
    """Apply standard tick styles to a Matplotlib graph.

    Args:
        ax (Axes): The graph to apply the styles to.

    Returns:
        Axes: The graph with the styles applied.
    """
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(GraphStyles.POPPINS_REG)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(GraphStyles.POPPINS_REG)

    return ax


def apply_hatches(ax: Axes, num_segments: int) -> Axes:
    """Apply hatch patterns to patches in a plot, such as bars, histograms, or area plots.

    This function divides the patches in the given Axes object into the specified
    number of segments and applies a different hatch pattern to each segment.

    Args:
        ax (Axes): The matplotlib Axes object containing the plot with patches (bars, histograms, etc.).
        num_segments (int): The number of segments to divide the patches into, with each segment receiving a different hatch pattern.

    Returns:
        Axes: The modified Axes object with hatches applied to the patches.
    """
    hatches = _hatches_gen()
    patch_groups = np.array_split(ax.patches, num_segments)
    for patch_group in patch_groups:
        hatch = next(hatches)
        for patch in patch_group:
            patch.set_hatch(hatch)
    return ax


def not_none(value1: any, value2: any) -> any:
    """Helper function that returns the first value that is not None.

    Args:
        value1: The first value.
        value2: The second value.

    Returns:
        The first value that is not None.
    """
    if value1 is None:
        return value2
    return value1


def get_decimals(ylim: tuple[float, float], tick_values: list[float], max_decimals: int = 10) -> int:
    """Helper function for the `human_format` function that determines the number of decimals to use for the y-axis.

    Args:
        ylim: The y-axis limits.
        tick_values: The y-axis tick values.
        max_decimals: The maximum number of decimals to use. Defaults to 100.

    Returns:
        int: The number of decimals to use.
    """
    decimals = 0
    while decimals < max_decimals:
        tick_labels = [human_format(t, 0, decimals=decimals) for t in tick_values if t >= ylim[0] and t <= ylim[1]]
        # Ensure no duplicate labels
        if len(tick_labels) == len(set(tick_labels)):
            break
        decimals += 1
    return decimals


def add_source_text(
    ax: Axes,
    source_text: str,
    font_size: float = GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE,
    vertical_padding: float = 2,
) -> Text:
    """Add source text to the bottom left corner of a graph.

    Args:
        ax (Axes): The graph to add the source text to.
        source_text (str): The source text.
        font_size (float, optional): The font size of the source text.
            Defaults to GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE.
        vertical_padding (float, optional): The padding in ems below the x-axis label. Defaults to 2.

    Returns:
        Text: The source text.
    """
    ax.figure.canvas.draw()

    # Get y coordinate of the text
    xlabel_box = ax.xaxis.label.get_window_extent(renderer=ax.figure.canvas.get_renderer())

    top_of_label_px = xlabel_box.y0

    padding_px = vertical_padding * font_size

    y_disp = top_of_label_px - padding_px - (xlabel_box.height)

    # Convert display coordinates to normalized figure coordinates
    y_norm = y_disp / ax.figure.bbox.height

    # Get x coordinate of the text
    ylabel_box = ax.yaxis.label.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    title_box = ax.title.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    min_x0 = min(ylabel_box.x0, title_box.x0)
    x_norm = ax.figure.transFigure.inverted().transform((min_x0, 0))[0]

    # Add text to the bottom left corner of the figure
    return ax.figure.text(
        x_norm,
        y_norm,
        source_text,
        ha="left",
        va="bottom",
        transform=ax.figure.transFigure,
        fontsize=GraphStyles.DEFAULT_SOURCE_FONT_SIZE,
        fontproperties=GraphStyles.POPPINS_LIGHT_ITALIC,
        color="dimgray",
    )
