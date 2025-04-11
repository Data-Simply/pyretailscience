"""Period on period module.

This module provides functionality for plotting multiple overlapping time periods
from the same time series on a single line chart using matplotlib.

The `plot` function is useful for visual comparisons of temporal trends
across different time windows, with each time window plotted as a separate line
but aligned to a common starting point.

Example use case: Comparing sales data across multiple promotional weeks or seasonal periods.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

import pyretailscience.style.graph_utils as gu
from pyretailscience.plots.line import plot

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


def overlapping_periods(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    periods: list[tuple[str, str]],
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

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        x_col (str): Name of the column representing datetime values.
        y_col (str): Name of the column representing the y-axis values (e.g. sales, counts).
        periods (List[Tuple[str, str]]): A list of (start_date, end_date) tuples
            representing the periods to plot.
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
        ValueError: If `periods` is an empty list.
    """
    if not periods:
        raise ValueError("The 'periods' list must contain at least one (start, end) tuple.")

    ax = ax or plt.gca()
    start_ref = pd.to_datetime(periods[0][0])

    df[x_col] = pd.to_datetime(df[x_col])
    for idx, (start_str, end_str) in enumerate(periods):
        style = LINE_STYLES[idx % len(LINE_STYLES)]
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        period_df = df[(df[x_col] >= start) & (df[x_col] <= end)].copy()

        if period_df.empty:
            continue

        time_offset = period_df[x_col].iloc[0] - start_ref
        period_df["realigned_date"] = period_df[x_col] - time_offset

        label = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        plot(
            df=period_df,
            x_col="realigned_date",
            value_col=y_col,
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
        y_label=y_label or y_col,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return ax
