"""This module provides functionality for creating timeline plots.

Which are essential for visualizing transactional data over time.
By aggregating data by specified periods (e.g., daily, weekly, monthly), timeline plots help to identify
trends, seasonal patterns, and performance variations across different timeframes. These plots are valuable tools for
retail analysis, sales tracking, and customer behavior insights.

### Features

- **Timeline Plot Creation**: Plot a value column (e.g., sales, transactions) over time, aggregated by a specific period (e.g., daily, weekly).
- **Customizable Aggregation**: Supports different aggregation functions (e.g., sum, average) to compute the value column's metrics.
- **Grouping by Categories**: Optionally group data by a specific category (e.g., product, region, store) and compare performance over time.
- **Time Period Handling**: The `period` parameter allows data aggregation by different time periods, such as days, weeks, or months.
- **Graph Styling**: Customize the appearance of the plot with options to adjust titles, axis labels, legend placement, and more.
- **Color Mapping**: Use linear color gradients for category-based groupings to visually differentiate between groups in the timeline.

### Use Cases

- **Sales and Revenue Analysis**: Track sales performance over time, either as a total or by group (e.g., product category or store).
- **Seasonal Trend Analysis**: Visualize how sales or transaction values fluctuate across different periods, helping to identify seasonal trends or promotional impacts.
- **Customer Behavior Tracking**: Examine changes in customer behavior (e.g., purchase frequency, average transaction value) over time.
- **Comparative Performance**: Compare multiple categories (e.g., different products or regions) on the same timeline to evaluate relative performance.

### Limitations and Handling of Data

- **Time Period Grouping**: Data is aggregated by a time period defined by the `period` argument, which can be adjusted to daily, weekly, monthly, etc.
- **Grouping by Categories**: If `group_col` is specified, the plot will display performance across different categories, with color differentiation for each group.
- **Flexible Aggregation**: The aggregation function (e.g., sum, average) can be customized to calculate the desired value for each period.

### Functionality Details

- **plot()**: Generates a timeline plot of a specified value column over time, with customization options for grouping, aggregation, and styling.
- **Helper functions**: Utilizes utility functions from the `pyretailscience` package to handle styling, formatting, and other plot adjustments.
"""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from pandas.tseries.offsets import BaseOffset

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import get_option
from pyretailscience.style.tailwind import COLORS, get_linear_cmap


def plot(
    df: pd.DataFrame,
    value_col: str,
    period: str | BaseOffset = "D",
    agg_func: str = "sum",
    group_col: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots the value_col over time.

    Timeline plots are a fundamental tool for interpreting transactional data within a temporal context. By presenting
    data in a chronological sequence, these visualizations reveal patterns and trends that might otherwise remain hidden
    in raw numbers, making them essential for both historical analysis and forward-looking insights. They are
    particularly useful for:

    - Tracking sales performance across different periods (e.g., daily, weekly, monthly)
    - Identifying seasonal patterns or promotional impacts on sales
    - Comparing the performance of different product categories or store locations over time
    - Visualizing customer behavior trends, such as purchase frequency or average transaction value

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
        legend_title (str, optional): The title of the legend. Defaults to None. When None the legend title is set to
            the title case of `group_col`
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    df["transaction_period"] = df[get_option("column.transaction_date")].dt.to_period(
        period,
    )

    if group_col is None:
        colors = COLORS["green"][500]
        df = df.groupby("transaction_period")[value_col].agg(agg_func)
        default_title = "Total Sales"
        show_legend = False
    else:
        colors = get_linear_cmap("green")(
            np.linspace(0, 1, df[group_col].nunique()),
        )
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
    ax = gu.standard_graph_styles(
        ax,
        title=gu.not_none(title, default_title),
        x_label=gu.not_none(x_label, ""),
        y_label=gu.not_none(y_label, value_col.title()),
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=show_legend,
    )

    decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())
    ax.yaxis.set_major_formatter(
        lambda x, pos: gu.human_format(x, pos, decimals=decimals),
    )

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    gu.standard_tick_styles(ax)

    return ax
