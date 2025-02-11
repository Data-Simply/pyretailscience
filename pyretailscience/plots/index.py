"""This module provides functionality for creating index plots in retail analytics.

Index plots are useful for comparing the performance of different categories or segments against a baseline or average,
typically set at 100. The module supports customization of the plot's appearance, sorting of data, and filtering by specific groups,
offering valuable insights into retail operations.


### Features

- **Index Plot Creation**: Visualize how categories or segments perform relative to a baseline value, typically set at 100.
  Useful for comparing performance across products, regions, or customer segments.
- **Flexible Sorting**: Sort data by either group or value to highlight specific trends in the data.
- **Data Filtering**: Filter data based on specified groups to focus on specific categories or exclude unwanted data.
- **Highlighting Range**: Highlight specific ranges of values (e.g., performance range between 80-120) to focus on performance.
- **Series Support**: Optionally include a `series_col` for plotting multiple series (e.g., time periods) within the same plot.
- **Graph Customization**: Adjust titles, axis labels, legend titles, and styling to match the specific context of the analysis.

### Use Cases

- **Retail Performance Comparison**: Compare product or regional performance to the company average or baseline using an index plot.
- **Customer Segment Analysis**: Evaluate customer segment behavior against overall performance, helping identify high-performing segments.
- **Operational Insights**: Identify areas of concern or opportunity by comparing store, region, or product performance against the baseline.
- **Visualizing Retail Strategy**: Support decision-making by visualizing which categories or products overperform or underperform relative to a baseline.

### Limitations and Handling of Data

- **Data Grouping and Aggregation**: Supports aggregation functions such as sum, average, etc., for calculating the index.
- **Sorting**: Sorting can be applied by group or value, allowing analysts to focus on specific trends. If `series_col` is provided, sorting by `group` is applied.
- **Group Filtering**: Users can exclude or include specific groups for focused analysis, with error handling to ensure conflicting options are not used simultaneously.

### Functionality Details

- **plot()**: Generates the index plot, which can be customized with multiple options such as sorting, filtering, and styling.
- **get_indexes()**: Helper function for calculating the index of the value column for a given subset of the dataframe based on filters and aggregation.

"""


from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.graph_utils import GraphStyles
from pyretailscience.style.tailwind import COLORS, get_linear_cmap

COLORMAP_MIN = 0.25
COLORMAP_MAX = 0.75


def plot(  # noqa: C901, PLR0913 (ignore complexity and line length)
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
        raise ValueError(
            "exclude_groups and include_only_groups cannot be used together.",
        )

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
            index_df = index_df.sort_values(
                by=group_col,
                ascending=sort_order == "ascending",
            )
        elif sort_by == "value":
            index_df = index_df.sort_values(
                by="index",
                ascending=sort_order == "ascending",
            )
    else:
        show_legend = True
        colors = get_linear_cmap("green")(
            np.linspace(COLORMAP_MIN, COLORMAP_MAX, df[series_col].nunique()),
        )

        if sort_by == "group":
            index_df = index_df.sort_values(
                by=[group_col, series_col],
                ascending=sort_order == "ascending",
            )
        index_df = index_df.pivot_table(
            index=group_col,
            columns=series_col,
            values="index",
            sort=False,
        )

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
    if highlight_range is not None:
        ax.axvline(
            highlight_range[0],
            color="black",
            linewidth=0.25,
            alpha=0.1,
            zorder=-1,
        )
        ax.axvline(
            highlight_range[1],
            color="black",
            linewidth=0.25,
            alpha=0.1,
            zorder=-1,
        )
        ax.axvspan(
            highlight_range[0],
            highlight_range[1],
            color="black",
            alpha=0.1,
            zorder=-1,
        )

    default_title = f"{value_col.title()} by {group_col.title()}"

    ax = gu.standard_graph_styles(
        ax=ax,
        title=gu.not_none(title, default_title),
        x_label=gu.not_none(x_label, "Index"),
        y_label=gu.not_none(y_label, group_col.title()),
        legend_title=legend_title,
        show_legend=show_legend,
    )

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
