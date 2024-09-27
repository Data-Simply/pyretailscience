"""This module provides flexible functionality for creating histograms from pandas DataFrames or Series.

It allows you to visualize distributions of one or more value columns and optionally group them by a categorical column.
The module is designed to handle both DataFrames and Series, allowing you to create simple histograms or compare
distributions across categories by splitting the data into multiple histograms.

### Core Features

- **Single or Multiple Histograms**: Plot one or more value columns (**`value_col`**) as histograms. For example, visualize
the distribution of a single metric or compare multiple metrics simultaneously.
- **Grouped Histograms**: Create separate histograms for each unique value in **`group_col`** (e.g., product categories
or regions), allowing for easy comparison of distributions across groups.
- **Range Clipping and Filling**: Use **`range_lower`** and **`range_upper`** to limit the values being plotted by
clipping them or filling values outside the range with **NaN**. This is particularly useful when visualizing specific
data ranges.
- **Comprehensive Customization**: Customize plot titles, axis labels, and legends, with the option to move the legend outside the plot.
- **Support for Pre-Aggregated Data**: The module assumes that the data has already been aggregated. No aggregation occurs during the plotting process.

### Use Cases

- **Distribution Analysis**: Visualize the distribution of key metrics like revenue, sales, or user activity using
single or multiple histograms.
- **Group Comparisons**: Compare distributions across different groups, such as product categories, geographic regions, or customer segments. For instance, plot histograms to show how sales vary across different product categories.
- **Trends and Ranges**: Use **range_lower** and **range_upper** to visualize data within specific ranges, filtering out
outliers or focusing on key metrics for analysis.

### Limitations and Handling of Data

- **Pre-Aggregated Data Required**: This module does not perform any data aggregation, so all data must be pre-aggregated before being passed in for plotting.
- **Grouped Histograms**: If **`group_col`** is provided, the data will be pivoted so that each unique value in
**`group_col`** becomes a separate histogram. Otherwise, a single histogram is plotted.
- **Series Support**: The module can also handle pandas Series, though **`group_col`** cannot be provided when plotting a Series.

### Additional Features

- **Range Clipping or Filling**: You can control how the data is visualized by specifying bounds. If data points fall
outside the defined range, you can either clip them to the boundary values or fill them with **NaN** for exclusion.
- **Legend Customization**: For multiple histograms, you can add legends, including the option to move the legend outside
the plot for clarity.

"""

from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib.colors import ListedColormap

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
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    range_lower: float | None = None,
    range_upper: float | None = None,
    range_method: Literal["clip", "fillna"] = "clip",
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots a histogram of `value_col`, optionally split by `group_col`.

    Args:
        df (pd.DataFrame | pd.Series): The dataframe (or series) to plot.
        value_col (str or list of str, optional): The column(s) to plot. Can be a list of columns for multiple histograms.
        group_col (str, optional): The column used to define different histograms.
        title (str, optional): The title of the plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        range_lower (float, optional): Lower bound for clipping or filling NA values.
        range_upper (float, optional): Upper bound for clipping or filling NA values.
        range_method (str, optional): Whether to "clip" values outside the range or "fillna". Defaults to "clip".
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    value_col = _prepare_value_col(df=df, value_col=value_col)

    if isinstance(df, pd.Series):
        df = df.to_frame(name=value_col[0])

    df = apply_range_clipping(
        df=df,
        value_col=value_col,
        range_lower=range_lower,
        range_upper=range_upper,
        range_method=range_method,
    )

    cmap = get_base_cmap()

    num_histograms = _get_num_histograms(df=df, value_col=value_col, group_col=group_col)

    ax = _plot_histogram(
        df=df,
        value_col=value_col,
        group_col=group_col,
        ax=ax,
        cmap=cmap,
        num_histograms=num_histograms,
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

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax=ax)


def _prepare_value_col(df: pd.DataFrame | pd.Series, value_col: str | list[str] | None) -> list[str]:
    """Ensures that value_col is properly handled and returned as a list.

    Args:
        df (pd.DataFrame | pd.Series): The input dataframe or series.
        value_col (str or list of str, optional): The column(s) to plot. If a single string, it is converted to a list.

    Returns:
        list[str]: The processed value_col as a list of strings.
    """
    if isinstance(df, pd.Series):
        return ["value"] if value_col is None else [value_col]

    if value_col is None:
        raise ValueError("Please provide a value column to plot.")

    if isinstance(value_col, str):
        value_col = [value_col]

    return value_col


def _prepare_dataframe(
    data: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None,
    group_col: str | None,
) -> pd.DataFrame:
    """Prepares and returns the dataframe with appropriate columns for plotting.

    Args:
        data (pd.DataFrame | pd.Series): The input dataframe or series to prepare.
        value_col (str or list of str, optional): The column(s) to plot. If a single string, it is converted to a list.
        group_col (str, optional): The column used to group data.

    Returns:
        pd.DataFrame: The prepared dataframe ready for plotting.
    """
    if isinstance(data, pd.Series):
        if group_col is not None:
            raise ValueError("When passing a Series, 'group_col' should not be provided.")
        value_col = ["value"] if value_col is None else [value_col]
        return data.to_frame(name=value_col[0])

    if value_col is None:
        raise ValueError("Please provide a value column to plot.")

    if isinstance(value_col, str):
        value_col = [value_col]

    return data


def apply_range_clipping(
    df: pd.DataFrame,
    value_col: list[str],
    range_lower: float | None,
    range_upper: float | None,
    range_method: Literal["clip", "fillna"],
) -> pd.DataFrame:
    """Applies range clipping or filling based on the provided method and returns the modified dataframe.

    Args:
        df (pd.DataFrame): The dataframe to apply range clipping to.
        value_col (list of str): The column(s) to apply clipping or filling to.
        range_lower (float, optional): Lower bound for clipping or filling NA values.
        range_upper (float, optional): Upper bound for clipping or filling NA values.
        range_method (Literal, optional): Whether to "clip" values outside the range or "fillna". Defaults to "clip".

    Returns:
        pd.DataFrame: The modified dataframe with the clipping or filling applied.
    """
    if range_lower is not None or range_upper is not None:
        if range_method == "clip":
            return df.assign(**{col: df[col].clip(lower=range_lower, upper=range_upper) for col in value_col})
        return df.assign(
            **{
                col: df[col].apply(
                    lambda x: np.nan
                    if (range_lower is not None and x < range_lower) or (range_upper is not None and x > range_upper)
                    else x,
                )
                for col in value_col
            },
        )

    return df


def _get_num_histograms(df: pd.DataFrame, value_col: list[str], group_col: str | None) -> int:
    """Calculates the number of histograms to be plotted.

    Args:
        df (pd.DataFrame): The dataframe being plotted.
        value_col (list of str): The column(s) being plotted.
        group_col (str, optional): The column used for grouping data into histograms.

    Returns:
        int: The number of histograms to plot.
    """
    num_histograms = len(value_col)

    if group_col is not None:
        num_histograms = max(num_histograms, df[group_col].nunique())

    return num_histograms


def _plot_histogram(
    df: pd.DataFrame,
    value_col: list[str],
    group_col: str | None,
    ax: Axes | None,
    cmap: ListedColormap,
    num_histograms: int,
    **kwargs: dict,
) -> Axes:
    """Plots histograms for the provided dataframe.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (list of str): The column(s) to plot.
        group_col (str, optional): The column used to group data into multiple histograms.
        ax (Axes, optional): Matplotlib axes object to plot on.
        cmap (ListedColormap): The colormap to use for the plot.
        num_histograms (int): The number of histograms being plotted.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        Axes: The matplotlib axes object with the plotted histogram.
    """
    add_legend = num_histograms > 1

    if group_col is None:
        return df[value_col].plot(kind="hist", ax=ax, alpha=0.5, legend=add_legend, color=cmap.colors[0], **kwargs)

    df_pivot = df.pivot(columns=group_col, values=value_col[0])

    # Plot all columns at once
    return df_pivot.plot(
        kind="hist",
        ax=ax,
        alpha=0.5,
        legend=add_legend,
        color=cmap.colors[: len(df_pivot.columns)],  # Use the appropriate number of colors
        **kwargs,
    )
