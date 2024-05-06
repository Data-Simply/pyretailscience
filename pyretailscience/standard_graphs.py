import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from pandas.tseries.offsets import BaseOffset

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.graph_utils import GraphStyles as gs
from pyretailscience.style.tailwind import COLORS, get_linear_cmap

# TODO: Consider simplifying this by reducing the color range in the get_linear_cmap function.
COLORMAP_MIN = 0.25
COLORMAP_MAX = 0.75


def time_plot(
    df: pd.DataFrame,
    value_col: str,
    period: str | BaseOffset = "D",
    agg_func: str = "sum",
    group_col: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: Axes | None = None,
    source_text: str = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """
    Plots the value_col over time.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str): The column to plot.
        period (str | BaseOffset): The period to group the data by.
        agg_func (str, optional): The aggregation function to apply to the value_col. Defaults to "sum".
        group_col (str, optional): The column to group the data by. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None. When None the title is set to
            `f"{value_col.title()} by {group_col.title()}"`
        x_label (str, optional): The x-axis label. Defaults to None. When None the x-axis label is set to blank
        y_label (str, optional): The y-axis label. Defaults to None. When None the y-axis label is set to the title
            case of `value_col`
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    df["transaction_period"] = df["transaction_datetime"].dt.to_period(period)

    if group_col is None:
        colors = COLORS["green"][500]
        df = df.groupby("transaction_period")[value_col].agg(agg_func)
        default_title = "Total Sales"
        show_legend = False
    else:
        colors = get_linear_cmap("green")(np.linspace(COLORMAP_MIN, COLORMAP_MAX, df[group_col].nunique()))
        df = (
            df.groupby([group_col, "transaction_period"])[value_col]
            .agg(agg_func)
            .reset_index()
            .pivot(index="transaction_period", columns=group_col, values=value_col)
        )
        default_title = f"{value_col.title()} by {group_col.title()}"
        show_legend = True

    ax = df.plot(
        linewidth=3,
        color=colors,
        legend=show_legend,
        ax=ax,
        **kwargs,
    )
    ax = gu.standard_graph_styles(ax)

    ax.set_title(
        gu.not_none(title, default_title),
        fontproperties=gs.POPPINS_SEMI_BOLD,
        fontsize=gs.DEFAULT_TITLE_FONT_SIZE,
    )
    ax.set_xlabel(
        gu.not_none(x_label, ""),
        fontproperties=gs.POPPINS_REG,
        fontsize=gs.DEFAULT_AXIS_LABEL_FONT_SIZE,
    )
    ax.set_ylabel(
        gu.not_none(y_label, value_col.title()),
        fontproperties=gs.POPPINS_REG,
        fontsize=gs.DEFAULT_AXIS_LABEL_FONT_SIZE,
    )

    decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())
    ax.yaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))

    if show_legend:
        legend = ax.legend(title="Segment", frameon=True)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

    if source_text is not None:
        ax.annotate(
            source_text,
            xy=(-0.1, -0.2),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=gs.DEFAULT_SOURCE_FONT_SIZE,
            fontproperties=gs.POPPINS_LIGHT_ITALIC,
            color="dimgray",
        )

    # Set the font properties for the tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(gs.POPPINS_REG)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(gs.POPPINS_REG)

    return ax
