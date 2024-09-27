"""This module contains functions for creating bar plots."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.tailwind import get_base_cmap


def plot(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None = None,
    group_col: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    bar_width: float = 0.35,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    orientation: Literal["horizontal", "h", "vertical", "v"] = "vertical",
    sort_order: Literal["ascending", "descending"] | None = None,
    data_label_format: Literal["absolute", "percentage"] | None = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Creates a bar plot with optional grouping, sorting, orientation, and data labels.

    Args:
        df (pd.DataFrame | pd.Series): The dataframe (or series) to plot.
        value_col (str or list of str, optional): The column(s) to plot.
        group_col (str, optional): The column used to define different bars.
        title (str, optional): The title of the plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        legend_title (str, optional): The title of the legend.
        bar_width (float, optional): The width of the bars. Defaults to 0.35.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        orientation (str, optional): Bar orientation ('vertical' or 'horizontal').
        sort_order (str, optional): Sort order ('ascending' or 'descending').
        data_label_format (str, optional): Data label format ('absolute' or 'percentage').
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    if isinstance(value_col, str):
        value_col = [value_col]

    if isinstance(df, pd.Series):
        df = df.to_frame(name=value_col[0])

    cmap = get_base_cmap()
    num_value_cols = len(value_col)
    plot_kind = "bar" if orientation in ["vertical", "v"] else "barh"

    # Handle sorting
    if sort_order is not None:
        df = df.sort_values(by=value_col[0], ascending=sort_order == "ascending")

    if group_col:
        indices = np.arange(len(df[group_col]))  # Group labels on x-axis

        if ax is None:
            fig, ax = plt.subplots()

        for i, col in enumerate(value_col):
            offset = i * bar_width  # Shift bars to avoid overlap
            ax.bar(indices + offset, df[col], width=bar_width, label=col, color=cmap.colors[i])

        ax.set_xticks(indices + bar_width / 2 * (num_value_cols - 1))  # Position ticks in the center of grouped bars
        ax.set_xticklabels(df[group_col])

    else:
        ax = df.plot(
            kind=plot_kind,
            y=value_col,
            ax=ax,
            color=cmap.colors[:num_value_cols],
            legend=(num_value_cols > 1),
            width=bar_width,
            **kwargs,
        )

    # Apply standard styles
    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )

    # Add data labels
    if data_label_format is not None:
        for container in ax.containers:
            if data_label_format == "absolute":
                labels = [f"{v.get_height():.0f}" if plot_kind == "bar" else f"{v.get_width():.0f}" for v in container]
            elif data_label_format == "percentage":
                labels = [
                    f"{(v.get_height() / df[value_col[0]].sum()):.1%}"
                    if plot_kind == "bar"
                    else f"{(v.get_width() / df[value_col[0]].sum()):.1%}"
                    for v in container
                ]
            ax.bar_label(container, labels=labels, label_type="edge")

    # Add source text
    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax=ax)
