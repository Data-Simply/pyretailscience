"""Helper functions for styling graphs."""

import importlib.resources as pkg_resources
from collections.abc import Generator
from itertools import cycle

import matplotlib.font_manager as fm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text

ASSETS_PATH = pkg_resources.files("pyretailscience").joinpath("assets")
_MAGNITUDE_SUFFIXES = ["", "K", "M", "B", "T", "P"]


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
    """Format a number in a human-readable format for Matplotlib, discarding trailing zeros.

    Args:
        num (float): The number to format.
        pos (int, optional): The position. Defaults to None. Only used for Matplotlib compatibility.
        decimals (int, optional): The number of decimals. Defaults to 0.
        prefix (str, optional): The prefix of the returned string, eg '$'. Defaults to "".

    Returns:
        str: The formatted number, with trailing zeros removed.
    """
    # The minimum difference between two numbers to receive a different suffix
    minimum_magnitude_difference = 1000.0
    magnitude = 0

    # Keep dividing by 1000 until the number is small enough
    while abs(num) >= minimum_magnitude_difference:
        magnitude += 1
        num /= minimum_magnitude_difference

    # Check if the number rounds to exactly 1000 at the current magnitude
    if round(abs(num), decimals) == minimum_magnitude_difference:
        num /= minimum_magnitude_difference
        magnitude += 1

    # If magnitude exceeds the predefined suffixes, continue with multiples of "P"
    if magnitude < len(_MAGNITUDE_SUFFIXES):
        suffix = _MAGNITUDE_SUFFIXES[magnitude]
    else:
        # Calculate how many times beyond "P" we've gone and append that to "P"
        extra_magnitude = magnitude - (len(_MAGNITUDE_SUFFIXES) - 1)
        suffix = f"{1000 ** extra_magnitude}P"

    # Format the number and remove trailing zeros
    formatted_num = f"{prefix}%.{decimals}f" % num
    formatted_num = formatted_num.rstrip("0").rstrip(".") if "." in formatted_num else formatted_num

    return f"{formatted_num}{suffix}"


def truncate_to_x_digits(num_str: str, digits: int) -> str:
    """Truncate a human-formatted number to the first `num_digits` significant digits.

    Args:
        num_str (str): The formatted number (e.g., '999.999K').
        digits (int): The number of digits to keep.

    Returns:
        str: The truncated formatted number (e.g., '999.9K').
    """
    # Split the number part and the suffix (e.g., "999.999K" -> "999.999" and "K")
    suffix = ""
    for s in _MAGNITUDE_SUFFIXES:
        if num_str.endswith(s) and s != "":
            suffix = s
            num_str = num_str[: -len(s)]  # Remove the suffix for now
            break

    # Handle negative numbers
    is_negative = num_str.startswith("-")
    if is_negative:
        num_str = num_str[1:]  # Remove the negative sign for now

    # Handle zero case explicitly
    if float(num_str) == 0:
        return f"0{suffix}"

    # Handle small numbers explicitly to avoid scientific notation
    scientific_notation_threshold = 1e-4
    if abs(float(num_str)) < scientific_notation_threshold:
        return f"{float(num_str):.{digits}f}".rstrip("0").rstrip(".")

    digits_before_decimal = len(num_str.split(".")[0])
    # Calculate how many digits to keep after the decimal
    digits_to_keep_after_decimal = digits - digits_before_decimal

    # Ensure truncation without rounding
    if digits_to_keep_after_decimal > 0:
        factor = 10**digits_to_keep_after_decimal
        truncated_num = str(int(float(num_str) * factor) / factor)
    else:
        factor = 10**digits
        truncated_num = str(int(float(num_str) * factor) / factor)

    # Reapply the negative sign if needed
    if is_negative:
        truncated_num = f"-{truncated_num}"

    # Remove unnecessary trailing zeros and decimal point
    truncated_num = truncated_num.rstrip("0").rstrip(".")

    return f"{truncated_num}{suffix}"


def _add_legend(ax: Axes, legend_title: str | None, move_legend_outside: bool) -> Axes:
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
    show_legend: bool = True,
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
        show_legend (bool): Whether to display the legend or not.

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

    if not show_legend:
        return ax

    return _add_legend(
        ax=ax,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )


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
    available_hatches = _hatches_gen()
    patch_groups = np.array_split(ax.patches, num_segments)
    for patch_group in patch_groups:
        hatch = next(available_hatches)
        for patch in patch_group:
            patch.set_hatch(hatch)

    legend = ax.get_legend()
    if legend:
        existing_hatches = [patch.get_hatch() for patch in ax.patches if patch.get_hatch() is not None]
        unique_hatches = [hatch for idx, hatch in enumerate(existing_hatches) if hatch not in existing_hatches[:idx]]
        for legend_patch, hatch in zip(legend.get_patches(), cycle(unique_hatches)):
            legend_patch.set_hatch(hatch)

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
