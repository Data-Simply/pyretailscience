"""Period on period module.

This module provides functionality for plotting multiple overlapping time periods
from the same time series on a single line chart using matplotlib.

The `plot` function is useful for visual comparisons of temporal trends
across different time windows, with each time window plotted as a separate line
but aligned to a common starting point.

Example use case: Comparing sales data across multiple promotional weeks or seasonal periods.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib.axes import Axes

import pyretailscience.style.graph_utils as gu
from pyretailscience.plots.line import plot as line_plot

LINE_STYLES = [
    "-",  # solid
    "--",  # dashed
    ":",  # dotted
    "-.",  # dashdot
    (0, (5, 10)),  # long dash with offset
    (0, (6, 6)),  # loosely dashed
    (0, (3, 5, 1, 5)),  # loosely dashdotted
    (0, (3, 5, 1, 5, 1, 5)),  # loosely dashdotdotted
]


def plot(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    periods: list[tuple[str | datetime, str | datetime]],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    ax: Axes | None = None,
    **kwargs: dict[str, any],
) -> Axes:
    """Plot multiple overlapping periods from a single time series as individual lines.

    This function is used to align and overlay several time intervals from the same
    dataset to facilitate visual comparison. Each period is realigned to the reference
    start date and plotted as a separate line using a distinct linestyle.

    Note:
        The `periods` argument accepts a list of (start_date, end_date) tuples,
        which define the time windows to overlay. Each element in the tuple can be either
        a string (e.g., "2022-01-01") or a `datetime` object. You can use
        `find_overlapping_periods` from `pyretailscience.utils.date` to generate
        the `periods` input automatically.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        x_col (str): Name of the column representing datetime values.
        value_col (str): Name of the column representing the y-axis values (e.g. sales, counts).
        periods (List[Tuple[Union[str, datetime], Union[str, datetime]]]):
            A list of (start_date, end_date) tuples representing the periods to plot.
        x_label (Optional[str]): Custom label for the x-axis.
        y_label (Optional[str]): Custom label for the y-axis.
        title (Optional[str]): Title for the plot.
        source_text (Optional[str]): Text to show below the plot as a data source.
        legend_title (Optional[str]): Title for the plot legend.
        move_legend_outside (bool): Whether to place the legend outside the plot area.
        ax (Optional[Axes]): Matplotlib Axes object to draw on. If None, a new one is created.
        **kwargs: Additional keyword arguments passed to the base line plot function.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object with the completed plot.

    Raises:
        ValueError: The 'periods' list must contain at least two (start, end) tuples for comparison.
    """
    min_period_length = 2
    if len(periods) < min_period_length:
        raise ValueError("The 'periods' list must contain at least two (start, end) tuples for comparison.")

    periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods]
    start_ref = periods[0][0]

    sorted_periods = sorted(periods, reverse=True, key=lambda x: pd.to_datetime(x[0]))

    ax = ax or plt.gca()

    period_styles = {period: LINE_STYLES[idx % len(LINE_STYLES)] for idx, period in enumerate(sorted_periods)}

    df[x_col] = pd.to_datetime(df[x_col])

    start_ref_year = start_ref.year

    for start_str, end_str in periods:
        style = period_styles[(start_str, end_str)]
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        period_df = df[(df[x_col] >= start) & (df[x_col] <= end)].copy()

        if period_df.empty:
            continue

        year_diff = start.year - start_ref_year

        period_df["realigned_date"] = period_df[x_col].apply(
            lambda d, year_diff=year_diff: d - relativedelta(years=year_diff),
        )

        label = f"{start_str.date()} to {end_str.date()}"
        line_plot(
            df=period_df,
            x_col="realigned_date",
            value_col=value_col,
            ax=ax,
            linestyle=style,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )
        line = ax.get_lines()[-1]
        line.set_label(label)
        line.set_linestyle(style)

    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label or x_col,
        y_label=y_label or value_col,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return ax
