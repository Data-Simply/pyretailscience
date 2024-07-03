"""Helper functions for styling graphs."""

import importlib.resources as pkg_resources

import matplotlib.font_manager as fm
from matplotlib.axes import Axes

ASSETS_PATH = pkg_resources.files("pyretailscience").joinpath("assets")


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


def standard_graph_styles(
    ax: Axes,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Axes:
    """Apply standard styles to a Matplotlib graph.

    Args:
        ax (Axes): The graph to apply the styles to.
        title (str, optional): The title of the graph. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None.
        y_label (str, optional): The y-axis label. Defaults to None.

    Returns:
        Axes: The graph with the styles applied.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", axis="x", color="#DAD8D7", alpha=0.5, zorder=1)
    ax.grid(which="major", axis="y", color="#DAD8D7", alpha=0.5, zorder=1)

    if title is not None:
        ax.set_title(
            title,
            fontproperties=GraphStyles.POPPINS_SEMI_BOLD,
            fontsize=GraphStyles.DEFAULT_TITLE_FONT_SIZE,
            pad=GraphStyles.DEFAULT_TITLE_PAD,
        )

    if x_label is not None:
        ax.set_xlabel(
            x_label,
            fontproperties=GraphStyles.POPPINS_REG,
            fontsize=GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE,
            labelpad=GraphStyles.DEFAULT_AXIS_LABEL_PAD,
        )

    if y_label is not None:
        ax.set_ylabel(
            y_label,
            fontproperties=GraphStyles.POPPINS_REG,
            fontsize=GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE,
            labelpad=GraphStyles.DEFAULT_AXIS_LABEL_PAD,
        )

    return ax


def not_none(value1: any, value2: any) -> any:
    """Helper funciont that returns the first value that is not None.

    Args:
        value1: The first value.
        value2: The second value.

    Returns:
        The first value that is not None.
    """
    if value1 is None:
        return value2
    return value1


def get_decimals(ylim: tuple[float, float], tick_values: list[float], max_decimals: int = 100) -> int:
    """Helper function for the `human_format` function that determines the number of decimals to use for the y-axis.

    Args:
        ylim: The y-axis limits.
        tick_values: The y-axis tick values.
        max_decimals: The maximum number of decimals to use. Defaults to 100.

    Returns:
        int: The number of decimals to use.
    """
    decimals = 0
    while True:
        tick_labels = [human_format(t, 0, decimals=decimals) for t in tick_values if t >= ylim[0] and t <= ylim[1]]
        if len(tick_labels) == len(set(tick_labels)) or decimals == max_decimals:
            break
        decimals += 1
    return decimals
