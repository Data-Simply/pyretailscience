"""This module provides flexible functionality for creating line plots from pandas DataFrames.

It focuses on visualizing sequences that are ordered or sequential but not necessarily categorical, such as "days since
an event" or "months since a competitor opened." However, while this module can handle datetime values on the x-axis,
the **plots.time_line** module has additional features that make working with datetimes easier, such as easily resampling the
data to alternate time frames.

The sequences used in this module can include values like "days since an event" (e.g., -2, -1, 0, 1, 2) or "months
since a competitor store opened." **This module is not intended for use with actual datetime values**. If a datetime
or datetime-like column is passed as **`x_col`**, a warning will be triggered, suggesting the use of the
**`plots.time_line`** module.

### Core Features

- **Plotting Sequences or Indexes**: Plot one or more value columns (**`value_col`**) with support for sequences like
-2, -1, 0, 1, 2 (e.g., months since an event), using either the index or a specified x-axis column (**`x_col`**).
- **Custom X-Axis or Index**: Use any column as the x-axis (**`x_col`**) or plot based on the index if no x-axis column is specified.
- **Multiple Lines**: Create separate lines for each unique value in **`group_col`** (e.g., categories or product types).
- **Comprehensive Customization**: Easily customize plot titles, axis labels, and legends, with the option to move the legend outside the plot.
- **Pre-Aggregated Data**: The data must be pre-aggregated before plotting, as no aggregation occurs within the module.

### Use Cases

- **Daily Trends**: Plot trends such as daily revenue or user activity, for example, tracking revenue since the start of the year.
- **Event Impact**: Visualize how metrics (e.g., revenue, sales, or traffic) change before and after an important event, such as a competitor store opening or a product launch.
- **Category Comparison**: Compare metrics across multiple categories over time, for example, tracking total revenue for the top categories before and after an event like the introduction of a new competitor.

### Limitations and Handling of Temporal Data

- **Limited Handling of Temporal Data**: This module can plot simple time-based sequences, such as "days since an event," but it cannot manipulate or directly handle datetime or date-like columns. It is not optimized for actual datetime values.
If a datetime column is passed or more complex temporal plotting is needed, a warning will suggest using the **`plots.time_line`** module, which is specifically designed for working with temporal data and performing time-based manipulation.
- **Pre-Aggregated Data Required**: The module does not perform any data aggregation, so all data must be pre-aggregated before being passed in for plotting.

"""

import warnings

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
    """Plots the `value_col` over the specified `x_col` or index, creating a separate line for each unique value in `group_col`.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str or list of str): The column(s) to plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different lines.
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
    if x_col is not None and pd.api.types.is_datetime64_any_dtype(df[x_col]):
        warnings.warn(
            f"The column '{x_col}' is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
            UserWarning,
            stacklevel=2,
        )

    elif x_col is None and pd.api.types.is_datetime64_any_dtype(df.index):
        warnings.warn(
            "The DataFrame index is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
            UserWarning,
            stacklevel=2,
        )
    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")
    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)

    is_multi_line = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    color_gen_threshold = 4
    num_colors = len(pivot_df.columns) if is_multi_line else 1
    color_gen = get_single_color_cmap() if num_colors < color_gen_threshold else get_multi_color_cmap()
    colors = [next(color_gen) for _ in range(num_colors)]

    ax = pivot_df.plot(
        ax=ax,
        linewidth=3,
        color=colors,
        legend=is_multi_line,
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
