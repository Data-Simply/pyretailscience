from matplotlib.axes import Axes


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
