"""This module provides functionality for creating broken timeline plots from pandas DataFrames.

A broken timeline plot visualizes data availability across categories over time, showing periods where
data is available as horizontal bars, with gaps indicating missing data periods.

### Features

- **Multiple Categories**: Support for displaying multiple categories with different colors
- **Customizable Periods**: Aggregate data by different time periods (daily, weekly)
- **Threshold Filtering**: Filter out values below a specified threshold
- **Date Formatting**: Uses matplotlib's ConciseDateFormatter for clean date axis labels

### Use Cases

- **Data Quality Assessment**: Visualize data availability gaps across categories/segments over time
- **Product Availability Analysis**: Identify periods with stock outs by store/category
- **Seasonality Analysis**: Assess to look for period of low sales that may indicate seasonality or other trends
"""

from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.options import get_option
from pyretailscience.plots.styles.tailwind import COLORS

# Period configurations: gap threshold and duration (in days)
PERIOD_CONFIG = {
    "D": 1,  # Daily: 1 day gap and duration
    "W": 7,  # Weekly: 1 week gap and duration
}


def _validate_inputs(df: pd.DataFrame, category_col: str, value_col: str, date_col: str, period: str) -> None:
    """Validate input parameters for the plot function.

    Args:
        df: Input DataFrame
        category_col: Category column name
        value_col: Value column name
        date_col: Date column name
        period: Time period for aggregation

    Raises:
        ValueError: If DataFrame is empty or invalid period specified
        KeyError: If required columns don't exist
    """
    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")

    # Validate required columns exist
    required_cols = [date_col, category_col, value_col]

    for col in required_cols:
        if col not in df.columns:
            msg = f"Required column '{col}' not found in DataFrame"
            raise KeyError(msg)

    # Validate period parameter
    valid_periods = list(PERIOD_CONFIG.keys())
    if period not in valid_periods:
        msg = f"Invalid period '{period}'. Must be one of {valid_periods}"
        raise ValueError(msg)


def plot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    period: str = "D",
    agg_func: str = "sum",
    threshold_value: float | None = None,
    bar_height: float = 0.8,
    figsize: tuple[int, int] | None = None,
    **kwargs: dict[str, Any],
) -> SubplotBase:
    """Creates a broken timeline plot showing data availability across categories over time.

    Shows periods where data is available as horizontal bars, with gaps indicating missing data periods.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be plotted.
        category_col (str): The column containing categories to display on y-axis.
        value_col (str): The column containing values to determine data availability.
        title (str, optional): The title of the plot. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to None.
        y_label (str, optional): The label for the y-axis. Defaults to None.
        ax (Axes, optional): The Matplotlib Axes object to plot on. Defaults to None.
        source_text (str, optional): Text to be displayed as a source at the bottom of the plot. Defaults to None.
        period (str, optional): Period for aggregating data using pandas to_period ("D", "W").
            Defaults to "D".
        agg_func (str, optional): The aggregation function to apply to the value_col when grouping by period.
            Defaults to "sum".
        threshold_value (float, optional): Values below this threshold are considered gaps. Defaults to None.
        bar_height (float, optional): Height of timeline bars as fraction of available space. Defaults to 0.8.
        figsize: tuple[int, int] | None = None,
        **kwargs (dict[str, Any]): Additional keyword arguments for matplotlib broken_barh function.

    Returns:
        SubplotBase: The Matplotlib Axes object with the generated plot.

    Raises:
        ValueError: If DataFrame is empty, required columns are missing, or invalid period specified.
        KeyError: If specified columns don't exist in the DataFrame.
    """
    date_col = get_option("column.transaction_date")

    # Convert period to uppercase to handle lowercase inputs
    period = period.upper()

    # Validate required columns exist
    _validate_inputs(df, category_col, value_col, date_col, period)

    # Create a copy of the data and ensure date column is datetime
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Apply threshold filter if specified
    if threshold_value is not None:
        df_copy = df_copy[df_copy[value_col] >= threshold_value]

    df_copy["period"] = df_copy[date_col].dt.to_period(period)
    df_copy = df_copy.groupby([category_col, "period"]).agg({value_col: agg_func}).reset_index()
    df_copy[date_col] = df_copy["period"].dt.start_time

    # Sort by date once for all categories
    df_copy = df_copy.sort_values(date_col)

    # Get unique categories and create y-axis mapping
    categories = sorted(df_copy[category_col].unique())
    category_to_y = {cat: i for i, cat in enumerate(categories)}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Use module-level period configuration
    gap_threshold = PERIOD_CONFIG[period]
    bar_color = kwargs.pop("color", COLORS["green"][500])

    # Process each category
    for category in categories:
        dates = df_copy[df_copy[category_col] == category][date_col].values

        # Convert to matplotlib date numbers and find segments
        dates_num = mdates.date2num(dates)
        gaps = np.diff(dates_num) > gap_threshold
        date_segments = np.split(dates_num, np.where(gaps)[0] + 1)

        # Calculate appropriate width based on period type
        base_width = PERIOD_CONFIG[period]
        segments = []
        for seg in date_segments:
            if len(seg) > 0:
                # Width should be number of periods * typical period duration
                width = len(seg) * base_width
                segments.append((seg[0], width))
        bar_offset = bar_height / 2
        ax.broken_barh(
            segments,
            (category_to_y[category] - bar_offset, bar_height),
            facecolors=bar_color,
            **kwargs,
        )

    # Configure y-axis
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.invert_yaxis()

    # Configure x-axis for dates
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Apply standard graph styles
    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    # Add source text if provided
    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax=ax)
