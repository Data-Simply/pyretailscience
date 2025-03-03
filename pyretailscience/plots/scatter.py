"""This module provides functionality for creating scatter plots from pandas DataFrames.

It is designed to visualize relationships between variables, highlight distributions, and compare different categories using scatter points.

### Core Features

- **Flexible X-Axis Handling**: Uses an index or a specified x-axis column (**`x_col`**) for plotting.
- **Multiple Scatter Groups**: Supports plotting multiple columns (**`value_col`**) or groups (**`group_col`**).
- **Dynamic Color Mapping**: Automatically selects a colormap based on the number of groups.
- **Legend Customization**: Supports custom legend titles and the option to move the legend outside the plot.
- **Source Text**: Provides an option to add source attribution to the plot.

### Use Cases

- **Category-Based Scatter Plots**: Compare different categories using scatter points.
- **Trend Analysis**: Identify patterns and outliers in datasets.
- **Multi-Value Scatter Plots**: Show multiple data series in a single scatter chart.

### Limitations and Warnings

- **Pre-Aggregated Data Required**: The module does not perform data aggregation; data should be pre-aggregated
before being passed to the function.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.tailwind import get_multi_color_cmap, get_single_color_cmap


def plot(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_col: str | None = None,
    group_col: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots a scatter chart for the given `value_col` over `x_col` or index, with optional grouping by `group_col`.

    Args:
        df (pd.DataFrame or pd.Series): The dataframe or series to plot.
        value_col (str or list of str): The column(s) to plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different scatter groups.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If `value_col` is a list and `group_col` is provided (which causes ambiguity in plotting).
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")

    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)

    is_multi_scatter = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    color_gen_threshold = 3
    num_colors = len(pivot_df.columns) if is_multi_scatter else 1
    color_gen = get_single_color_cmap() if num_colors < color_gen_threshold else get_multi_color_cmap()
    colors = [next(color_gen) for _ in range(num_colors)]

    ax = ax or plt.gca()
    alpha = kwargs.pop("alpha", 0.7)
    for col, color in zip(pivot_df.columns, colors, strict=False):
        ax.scatter(
            pivot_df.index,
            pivot_df[col],
            color=color,
            label=col if is_multi_scatter else None,
            alpha=alpha,
            **kwargs,
        )

    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax)
