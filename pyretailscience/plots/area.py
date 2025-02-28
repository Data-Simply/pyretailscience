"""This module provides functionality for creating area plots from pandas DataFrames.

It is designed to visualize data distributions over time or across categories using filled area charts. These plots
help highlight trends and comparisons between different groups by stacking or overlaying areas.

While this module supports datetime values on the x-axis, the **plots.time_area** module is better suited for
explicitly time-based visualizations, offering features like resampling and time-based aggregation.

### Core Features

- **Flexible X-Axis Handling**: Uses an index or a specified x-axis column (**`x_col`**) for plotting.
- **Multiple Area Support**: Allows plotting multiple columns (**`value_col`**) or groups (**`group_col`**).
- **Dynamic Color Mapping**: Automatically selects a colormap based on the number of groups.
- **Legend Customization**: Supports custom legend titles and the option to move the legend outside the plot.
- **Source Text**: Provides an option to add source attribution to the plot.

### Use Cases

- **Time Series Visualization**: Show trends in a metric over time (e.g., revenue by month).
- **Stacked Area Charts**: Compare contributions of different groups over time.
- **Category-Based Area Plots**: Visualize distributions of data across categories.

### Limitations and Warnings

- **Handling of Datetime Data**: If a datetime column is passed as **`x_col`**, a warning suggests using
  the **plots.time_area** module for better handling.
- **Pre-Aggregated Data Required**: The module does not perform data aggregation; data should be pre-aggregated
  before being passed to the function.
"""

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.tailwind import get_multi_color_cmap, get_single_color_cmap


def plot(
    df: pd.DataFrame,
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
    """Plots an area chart for the given `value_col` over `x_col` or index, with optional grouping by `group_col`.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str or list of str): The column(s) to plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different areas in the plot.
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

    is_multi_area = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    color_gen_threshold = 4
    num_colors = len(pivot_df.columns) if is_multi_area else 1
    color_gen = get_single_color_cmap() if num_colors < color_gen_threshold else get_multi_color_cmap()
    colors = [next(color_gen) for _ in range(num_colors)]
    alpha = kwargs.pop("alpha", 0.7)
    ax = pivot_df.plot(
        ax=ax,
        kind="area",
        alpha=alpha,
        color=colors,
        legend=is_multi_area,
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
