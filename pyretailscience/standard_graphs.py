"""This module contains functions to create standard graphs for retail science projects."""

from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from pandas.tseries.offsets import BaseOffset

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import get_option
from pyretailscience.style.graph_utils import GraphStyles
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
    df["transaction_period"] = df[get_option("column.transaction_date")].dt.to_period(period)

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
    ax = gu.standard_graph_styles(
        ax,
        title=gu.not_none(title, default_title),
        x_label=gu.not_none(x_label, ""),
        y_label=gu.not_none(y_label, value_col.title()),
    )

    decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())
    ax.yaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))

    if show_legend:
        legend_bbox_to_anchor = None
        if move_legend_outside:
            legend_bbox_to_anchor = (1.05, 1)
        legend = ax.legend(
            title=gu.not_none(legend_title, group_col.title()),
            frameon=True,
            bbox_to_anchor=legend_bbox_to_anchor,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    gu.standard_tick_styles(ax)

    return ax


def get_indexes(
    df: pd.DataFrame,
    df_index_filter: list[bool],
    index_col: str,
    value_col: str,
    index_subgroup_col: str | None = None,
    agg_func: str = "sum",
    offset: int = 0,
) -> pd.DataFrame:
    """Calculates the index of the value_col for the subset of a dataframe defined by df_index_filter.

    Args:
        df (pd.DataFrame): The dataframe to calculate the index on.
        df_index_filter (list[bool]): The boolean index to filter the data by.
        index_col (str): The column to calculate the index on.
        value_col (str): The column to calculate the index on.
        index_subgroup_col (str, optional): The column to subgroup the index by. Defaults to None.
        agg_func (str): The aggregation function to apply to the value_col.
        offset (int, optional): The offset to subtract from the index. Defaults to 0.

    Returns:
        pd.Series: The index of the value_col for the subset of data defined by filter_index.
    """
    if all(df_index_filter) or not any(df_index_filter):
        raise ValueError("The df_index_filter cannot be all True or all False.")

    grp_cols = [index_col] if index_subgroup_col is None else [index_subgroup_col, index_col]

    overall_df = df.groupby(grp_cols)[value_col].agg(agg_func).to_frame(value_col)
    if index_subgroup_col is None:
        overall_total = overall_df[value_col].sum()
    else:
        overall_total = overall_df.groupby(index_subgroup_col)[value_col].sum()
    overall_s = overall_df[value_col] / overall_total

    subset_df = df[df_index_filter].groupby(grp_cols)[value_col].agg(agg_func).to_frame(value_col)
    if index_subgroup_col is None:
        subset_total = subset_df[value_col].sum()
    else:
        subset_total = subset_df.groupby(index_subgroup_col)[value_col].sum()
    subset_s = subset_df[value_col] / subset_total

    return ((subset_s / overall_s * 100) - offset).to_frame("index").reset_index()


# TODO: Refactor this into a class to reduce complexity and arg count
def index_plot(  # noqa: C901, PLR0913 (ignore complexity and line length)
    df: pd.DataFrame,
    df_index_filter: list[bool],
    value_col: str,
    group_col: str,
    agg_func: str = "sum",
    series_col: str | None = None,
    title: str | None = None,
    x_label: str = "Index",
    y_label: str | None = None,
    legend_title: str | None = None,
    highlight_range: Literal["default"] | tuple[float, float] | None = "default",
    sort_by: Literal["group", "value"] | None = "group",
    sort_order: Literal["ascending", "descending"] = "ascending",
    ax: Axes | None = None,
    source_text: str | None = None,
    exclude_groups: list[any] | None = None,
    include_only_groups: list[any] | None = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Creates an index plot.

    Index plots are visual tools used in retail analytics to compare different categories or segments against a
    baseline or average value, typically set at 100. Index plots allow analysts to:

    Quickly identify which categories over- or underperform relative to the average
    Compare performance across diverse categories on a standardized scale
    Highlight areas of opportunity or concern in retail operations
    Easily communicate relative performance to stakeholders without revealing sensitive absolute numbers

    In retail contexts, index plots are valuable for:

    Comparing sales performance across product categories
    Analyzing customer segment behavior against the overall average
    Evaluating store or regional performance relative to company-wide metrics
    Identifying high-potential areas for growth or investment

    By normalizing data to an index, these plots facilitate meaningful comparisons and help focus attention on
    significant deviations from expected performance, supporting more informed decision-making in retail strategy and
    operations.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        df_index_filter (list[bool]): The filter to apply to the dataframe.
        value_col (str): The column to plot.
        group_col (str): The column to group the data by.
        agg_func (str, optional): The aggregation function to apply to the value_col. Defaults to "sum".
        series_col (str, optional): The column to use as the series. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None. When None the title is set to
            `f"{value_col.title()} by {group_col.title()}"`
        x_label (str, optional): The x-axis label. Defaults to "Index".
        y_label (str, optional): The y-axis label. Defaults to None. When None the y-axis label is set to the title
            case of `group_col`
        legend_title (str, optional): The title of the legend. Defaults to None. When None the legend title is set to
            the title case of `group_col`
        highlight_range (Literal["default"] | tuple[float, float] | None, optional): The range to highlight. Defaults
            to "default". When "default" the range is set to (80, 120). When None no range is highlighted.
        sort_by (Literal["group", "value"] | None, optional): The column to sort by. Defaults to "group". When None the
            data is not sorted. When "group" the data is sorted by group_col. When "value" the data is sorted by
            the value_col. When series_col is not None this option is ignored.
        sort_order (Literal["ascending", "descending"], optional): The order to sort the data. Defaults to "ascending".
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        exclude_groups (list[any], optional): The groups to exclude from the plot. Defaults to None.
        include_only_groups (list[any], optional): The groups to include in the plot. Defaults to None. When None all
            groups are included. When not None only the groups in the list are included. Can not be used with
            exclude_groups.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If sort_by is not either "group" or "value" or None.
        ValueError: If sort_order is not either "ascending" or "descending".
        ValueError: If exclude_groups and include_only_groups are used together.
    """
    if sort_by is not None and sort_by not in ["group", "value"]:
        raise ValueError("sort_by must be either 'group' or 'value' or None")
    if sort_order not in ["ascending", "descending"]:
        raise ValueError("sort_order must be either 'ascending' or 'descending'")
    if exclude_groups is not None and include_only_groups is not None:
        raise ValueError("exclude_groups and include_only_groups cannot be used together.")

    index_df = get_indexes(
        df=df,
        df_index_filter=df_index_filter,
        index_col=group_col,
        index_subgroup_col=series_col,
        value_col=value_col,
        agg_func=agg_func,
        offset=100,
    )

    if exclude_groups is not None:
        index_df = index_df[~index_df[group_col].isin(exclude_groups)]
    if include_only_groups is not None:
        index_df = index_df[index_df[group_col].isin(include_only_groups)]

    if series_col is None:
        colors = COLORS["green"][500]
        show_legend = False
        index_df = index_df[[group_col, "index"]].set_index(group_col)
        if sort_by == "group":
            index_df = index_df.sort_values(by=group_col, ascending=sort_order == "ascending")
        elif sort_by == "value":
            index_df = index_df.sort_values(by="index", ascending=sort_order == "ascending")
    else:
        show_legend = True
        colors = get_linear_cmap("green")(np.linspace(COLORMAP_MIN, COLORMAP_MAX, df[series_col].nunique()))

        if sort_by == "group":
            index_df = index_df.sort_values(by=[group_col, series_col], ascending=sort_order == "ascending")
        index_df = index_df.pivot_table(index=group_col, columns=series_col, values="index", sort=False)

    ax = index_df.plot.barh(
        left=100,
        legend=show_legend,
        ax=ax,
        color=colors,
        width=GraphStyles.DEFAULT_BAR_WIDTH,
        zorder=2,
        **kwargs,
    )

    ax.axvline(100, color="black", linewidth=1, alpha=0.5)
    if highlight_range == "default":
        highlight_range = (80, 120)
    elif highlight_range is not None:
        ax.axvline(highlight_range[0], color="black", linewidth=0.25, alpha=0.1, zorder=-1)
        ax.axvline(highlight_range[1], color="black", linewidth=0.25, alpha=0.1, zorder=-1)
        ax.axvspan(highlight_range[0], highlight_range[1], color="black", alpha=0.1, zorder=-1)

    default_title = f"{value_col.title()} by {group_col.title()}"

    ax = gu.standard_graph_styles(
        ax=ax,
        title=gu.not_none(title, default_title),
        x_label=gu.not_none(x_label, "Index"),
        y_label=gu.not_none(y_label, group_col.title()),
    )

    if show_legend:
        legend = ax.legend(title=gu.not_none(legend_title, series_col.title()), frameon=True)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    gu.standard_tick_styles(ax)

    return ax


def waterfall_plot(
    amounts: list[float],
    labels: list[str],
    title: str | None = None,
    y_label: str | None = None,
    x_label: str = "",
    source_text: str | None = None,
    data_label_format: Literal["absolute", "percentage", "both"] | None = None,
    display_net_bar: bool = False,
    display_net_line: bool = False,
    remove_zero_amounts: bool = True,
    ax: Axes | None = None,
    **kwargs: dict[str, any],
) -> Axes:
    """Generates a waterfall chart.

    Waterfall plots are particularly good for showing how different things add or subtract from a starting number. For
    instance,

    - Changes in sales figures from one period to another
    - Breakdown of profit margins
    - Impact of different product categories on overall revenue

    They are often used to identify key drivers of financial performance, highlight areas for improvement, and communicate
    complex data stories to stakeholders in an intuitive manner.

    Args:
        amounts (list[float]): The amounts to plot.
        labels (list[str]): The labels for the amounts.
        title (str, optional): The title of the chart. Defaults to None.
        y_label (str, optional): The y-axis label. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        data_label_format (Literal["absolute", "percentage", "both", "none"], optional): The format of the data labels.
            Defaults to "absolute".
        display_net_bar (bool, optional): Whether to display a net bar. Defaults to False.
        display_net_line (bool, optional): Whether to display a net line. Defaults to False.
        remove_zero_amounts (bool, optional): Whether to remove zero amounts from the plot. Defaults to True
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function.

    Returns:
        Axes: The matplotlib axes object.
    """
    if len(amounts) != len(labels):
        raise ValueError("The lengths of amounts and labels must be the same.")

    data_label_format = None if data_label_format is None else data_label_format.lower()
    if data_label_format is not None and data_label_format not in ["absolute", "percentage", "both"]:
        raise ValueError("data_label_format must be either 'absolute', 'percentage', 'both', or None.")

    df = pd.DataFrame({"labels": labels, "amounts": amounts})

    if remove_zero_amounts:
        df = df[df["amounts"] != 0]

    amount_total = df["amounts"].sum()

    colors = df["amounts"].apply(lambda x: COLORS["green"][500] if x > 0 else COLORS["red"][500]).to_list()
    bottom = df["amounts"].cumsum().shift(1).fillna(0).to_list()

    if display_net_bar:
        # Append a row for the net amount
        df.loc[len(df)] = ["Net", amount_total]
        colors.append(COLORS["blue"][500])
        bottom.append(0)

    # Create the plot
    ax = df.plot.bar(
        x="labels",
        y="amounts",
        legend=None,
        bottom=bottom,
        color=colors,
        width=0.8,
        ax=ax,
        **kwargs,
    )

    extra_title_pad = 25 if data_label_format != "none" else 0
    ax = gu.standard_graph_styles(
        ax,
        title=title,
        y_label=gu.not_none(y_label, "Amounts"),
        x_label=x_label,
        title_pad=GraphStyles.DEFAULT_TITLE_PAD + extra_title_pad,
    )

    decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())
    ax.yaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))

    # Add a black line at the y=0 position
    ax.axhline(y=0, color="black", linewidth=1, zorder=-1)

    if data_label_format is not None:
        total_change = df["amounts"].sum()
        if data_label_format == "absolute":
            labels = df["amounts"].apply(lambda x: gu.human_format(x, decimals=decimals + 1))
        elif data_label_format == "percentage":
            labels = df["amounts"].apply(lambda x: f"{x/total_change:.0%}")
        else:
            labels = [f"{gu.human_format(x, decimals=decimals+1)} ({x/total_change:.0%})" for x in df["amounts"]]

        ax.bar_label(
            ax.containers[0],
            label_type="edge",
            labels=labels,
            padding=5,
            fontsize=GraphStyles.DEFAULT_BAR_LABEL_FONT_SIZE,
            fontproperties=GraphStyles.POPPINS_REG,
        )

    if display_net_line:
        ax.axhline(y=amount_total, color="black", linewidth=1, linestyle="--")

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    gu.standard_tick_styles(ax)

    return ax
