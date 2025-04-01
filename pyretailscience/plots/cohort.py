"""This module provides functionality for creating cohort plots from pandas DataFrames.

It is designed to visualize data distributions using color-coded heatmaps, helping to highlight trends and comparisons between different groups.

### Core Features

- **Color Mapping**: Uses a predefined colormap for visualizing data.
- **Customizable Labels**: Supports custom labels for x-axis, y-axis, title, and colorbar.
- **Source Text**: Provides an option to add source attribution to the plot.
- **Grid and Tick Customization**: Applies standard styling for better readability.

### Use Cases

- **Cohort Analysis**: Visualize how different groups behave over time.
- **Category-Based Heatmaps**: Compare values across different categories.

### Limitations and Warnings

- **Data Aggregation Required**: The module does not perform data aggregation; data should be pre-aggregated before being passed to the function.
- **Fixed Color Mapping**: The module uses a predefined colormap without dynamic adjustments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.tailwind import get_listed_cmap


def plot(
    df: pd.DataFrame,
    cbar_label: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    percentage: bool = True,
    figsize: tuple[int, int] | None = None,
    **kwargs: dict,
) -> SubplotBase:
    """Plots a cohort plot for the given DataFrame.

    Args:
        df (pd.DataFrame): Dataframe containing cohort analysis data.
        cbar_label (str): Label for the colorbar.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): Additional source text annotation.
        percentage (bool, optional): If True, displays cohort values as percentages. Defaults to False.
        figsize (tuple[int, int], optional): The size of the plot. Defaults to None.
        **kwargs: Additional keyword arguments for cohort styling.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    cmap = get_listed_cmap("green")
    im = ax.imshow(df, cmap=cmap, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, format=ticker.StrMethodFormatter("{x:.0%}" if percentage else "{x:,.0f}"))
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize="x-large")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels(df.columns, rotation_mode="anchor")
    ax.set_yticklabels(df.index.astype(str))

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticks(np.arange(df.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(df.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    threshold = im.norm(1.0) / 2.0 if percentage else im.norm(df.to_numpy().max()) / 2.0
    valfmt = ticker.StrMethodFormatter("{x:.0%}" if percentage else "{x:,.0f}")
    textcolors = ("black", "white")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            color = textcolors[int(im.norm(df.iloc[i, j]) > threshold)]
            ax.text(j, i, valfmt(df.iloc[i, j], None), ha="center", va="center", color=color, fontsize=7)

    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )
    ax.grid(False)
    ax.hlines(y=3 - 0.5, xmin=-0.5, xmax=df.shape[1] - 0.5, color="white", linewidth=4)

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax)
