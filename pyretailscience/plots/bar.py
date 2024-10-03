"""This module provides flexible functionality for creating bar plots from pandas DataFrames or Series.

It allows you to create bar plots with optional grouping, sorting, orientation, and data labels. The module supports
both single and grouped bar plots, where grouped bars are created by providing a grouping column.

### Core Features

- **Single or Grouped Bar Plots**: Plot one or more value columns (`value_col`) as bars. Grouped bars are created by
  specifying both `value_col` (list of columns) and `group_col` (column used for grouping bars).
- **Sorting and Orientation**: Customize the sorting of bars (ascending or descending) and choose between vertical or
  horizontal bar orientation.
- **Data Labels**: Add data labels to bars, either showing absolute values or percentages.
- **Hatching Patterns**: Apply hatch patterns to the bars for enhanced visual differentiation.
- **Legend Customization**: Move the legend outside the plot for better visibility, especially when dealing with grouped bars.

### Use Cases

- **Sales and Revenue Analysis**: Visualize sales or revenue across different products or categories by creating grouped
  bar plots (e.g., revenue across quarters or regions).
- **Comparative Analysis**: Compare multiple metrics simultaneously by plotting grouped bars. For instance, you can
  compare product sales for different periods side by side.
- **Distribution Analysis**: Visualize the distribution of categorical data (e.g., product sales) across different
  categories.

### Limitations and Handling of Data

- **Pre-Aggregated Data Required**: The module does not perform any data aggregation, so all data must be pre-aggregated before
  being passed in for plotting.
- **Grouped Bars**: When `group_col` is provided, it will group the data and create a set of bars for each group.

### Additional Features

- **Data Label Customization**: Show either absolute values or percentages as labels for the bars.
- **Legend Customization**: For multiple value columns, you can move the legend outside the plot.
- **Orientation**: Choose between vertical (`"v"`, `"vertical"`) and horizontal (`"h"`, `"horizontal"`) bar orientations.
"""

from typing import Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib.patches import Rectangle

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.tailwind import get_multi_color_cmap, get_single_color_cmap


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
    use_hatch: bool = False,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Creates a customizable bar plot from a DataFrame or Series with optional features like sorting, orientation, and adding data labels. Grouped bars can be created with the use of a grouping column.

    Args:
        df (pd.DataFrame | pd.Series): The input DataFrame or Series containing the data to be plotted.
        value_col (str | list[str], optional): The column(s) containing values to plot as bars. Multiple value columns
                                               create grouped bars. Defaults to None.
        group_col (str, optional): The column to group data by, used for grouping bars. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to None.
        y_label (str, optional): The label for the y-axis. Defaults to None.
        legend_title (str, optional): The title for the legend. Defaults to None.
        bar_width (float, optional): The width of the bars. Defaults to 0.35.
        ax (Axes, optional): The Matplotlib Axes object to plot on. Defaults to None.
        source_text (str, optional): Text to be displayed as a source at the bottom of the plot. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area. Defaults to False.
        orientation (Literal["horizontal", "h", "vertical", "v"], optional): Orientation of the bars. Can be
                                                                             "horizontal", "h", "vertical", or "v".
                                                                             Defaults to "vertical".
        sort_order (Literal["ascending", "descending"] | None, optional): Sorting order for the bars. Can be
                                                                          "ascending" or "descending". Defaults to None.
        data_label_format (Literal["absolute", "percentage"] | None, optional): Format for displaying data labels.
                                                                                "absolute" shows raw values,
                                                                                "percentage" shows percentage.
                                                                                Defaults to None.
        use_hatch (bool, optional): Whether to apply hatch patterns to the bars. Defaults to False.
        **kwargs (dict[str, any]): Additional keyword arguments for the Pandas `plot` function.

    Returns:
        SubplotBase: The Matplotlib Axes object with the generated plot.
    """
    valid_orientations = ["horizontal", "h", "vertical", "v"]
    if orientation not in valid_orientations:
        error_msg = f"Invalid orientation: {orientation}. Expected one of {valid_orientations}"
        raise ValueError(error_msg)

    # Validate the sort_order value
    valid_sort_orders = ["ascending", "descending", None]
    if sort_order not in valid_sort_orders:
        error_msg = f"Invalid sort_order: {sort_order}. Expected one of {valid_sort_orders}"
        raise ValueError(error_msg)

    # Validate the data_label_format value
    valid_data_label_formats = ["absolute", "percentage", None]
    if data_label_format not in valid_data_label_formats:
        error_msg = f"Invalid data_label_format: {data_label_format}. Expected one of {valid_data_label_formats}"
        raise ValueError(error_msg)

    value_col = [value_col] if isinstance(value_col, str) else (["Value"] if value_col is None else value_col)

    df = df.to_frame(name=value_col[0]) if isinstance(df, pd.Series) else df

    df = df.sort_values(by=value_col[0], ascending=sort_order == "ascending") if sort_order is not None else df

    cmap = get_multi_color_cmap() if len(value_col) > 1 else get_single_color_cmap()

    plot_kind = "bar" if orientation in ["vertical", "v"] else "barh"

    ax = df.plot(
        kind=plot_kind,
        y=value_col,
        x=group_col,
        ax=ax,
        color=[next(cmap) for _ in range(len(value_col))],
        legend=(len(value_col) > 1),
        width=bar_width,
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

    if use_hatch:
        ax = gu.apply_hatches(ax=ax, num_segments=len(value_col))

    # Add data labels
    if data_label_format:
        _generate_bar_labels(
            ax=ax,
            plot_kind=plot_kind,
            value_col=value_col,
            df=df,
            data_label_format=data_label_format,
        )

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax=ax)


def _generate_bar_labels(
    ax: Axes,
    plot_kind: str,
    value_col: list[str],
    df: pd.DataFrame,
    data_label_format: Literal["absolute", "percentage"],
) -> None:
    """Adds bar labels to the bar plot containers.

    Args:
        ax (Axes): The matplotlib axes object containing the plot.
        plot_kind (str): The type of bar plot ('bar' for vertical or 'barh' for horizontal).
        value_col (list[str]): List of value columns for the bars.
        df (pd.DataFrame): Dataframe containing the data used in the plot.
        data_label_format (str): The format for displaying data labels ('absolute' or 'percentage').
    """

    def get_bar_value(v: Rectangle) -> float:
        return v.get_height() if plot_kind == "bar" else v.get_width()

    total_sum = df[value_col[0]].sum()  # Precompute sum for percentage calculations
    for container in ax.containers:
        if data_label_format == "absolute":
            labels = [f"{get_bar_value(v):.0f}" for v in container]
        elif data_label_format == "percentage":
            labels = [f"{(get_bar_value(v) / total_sum):.1%}" for v in container]
        ax.bar_label(container, labels=labels, label_type="edge")
