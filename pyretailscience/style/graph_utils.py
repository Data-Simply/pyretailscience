from matplotlib.axes import Axes


class GraphStyles:
    """A class to hold the styles for a graph."""

    DEFAULT_TITLE_FONT_SIZE = 16
    DEFAULT_SOURCE_FONT_SIZE = 10
    DEFAULT_AXIS_LABEL_FONT_SIZE = 12
    DEFAULT_TICK_LABEL_FONT_SIZE = 10


def human_format(num, pos=None, decimals=0, prefix="") -> str:
    """Format a number in a human readable format for Matplotlib.

    Args:
        num (float): The number to format.
        pos (int, optional): The position. Defaults to None. Only used for Matplotlib compatibility.
        decimals (int, optional): The number of decimals. Defaults to 0.
        prefix (str, optional): The prefix of the returned string, eg '$'. Defaults to "".

    Returns:
        str: The formatted number.
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    # Add more suffixes if you need them
    return f"{prefix}%.{decimals}f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


def standard_graph_styles(ax: Axes) -> Axes:
    """Apply standard styles to a Matplotlib graph.

    Args:
        ax (Axes): The graph to apply the styles to.

    Returns:
        Axes: The graph with the styles applied.
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", axis="x", color="#DAD8D7", alpha=0.5, zorder=1)
    ax.grid(which="major", axis="y", color="#DAD8D7", alpha=0.5, zorder=1)
    return ax


def not_none(value1, value2):
    """
    Helper funciont that returns the first value that is not None.

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
    """
    Helper function for the `human_format` function that determines the number of decimals to use for the y-axis.

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
