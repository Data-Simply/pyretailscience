"""This module provides functionality to generate a waterfall chart.

A visualization commonly used to illustrate how
different positive and negative values contribute to a cumulative total.Waterfall charts are effective in showing
the incremental impact of individual components, making them particularly useful for financial analysis,
performance tracking, and visualizing changes over time.

### Features

- **Waterfall Chart Creation**: Displays how different positive and negative values affect a starting total.
- **Data Label Formatting**: Supports custom formatting for data labels, including absolute values, percentages, or both.
- **Net Line and Bar Display**: Optionally includes a net line and net bar to show the overall cumulative result.
- **Customizable Plot Style**: Options to customize chart titles, axis labels, and remove zero amounts for better clarity.
- **Handling of Zero Amounts**: Allows removal of zero amounts from the plot to avoid cluttering the chart.
- **Interactive Elements**: Supports custom annotations for the chart with source text.

### Use Cases

- **Financial Analysis**: Show the breakdown of profits and losses over multiple periods, or how different cost categories affect overall margin.
- **Revenue Tracking**: Track how revenue or other key metrics change over time, and visualize the impact of individual contributing factors.
- **Performance Visualization**: Highlight how various business or product categories affect overall performance, such as sales, expenses, or growth metrics.
- **Budget Breakdown**: Visualize how different spending categories contribute to a total budget over a period.

### Functionality Details

- **plot()**: Generates a waterfall chart from a list of amounts and labels. It supports additional customization for display settings, labels, and source text.
- **format_data_labels()**: A helper function used to format the data labels according to the specified format (absolute, percentage, both).
"""

from typing import Literal

import pandas as pd
from matplotlib.axes import Axes

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.graph_utils import GraphStyles
from pyretailscience.style.tailwind import COLORS


def plot(
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
    instance:
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

    data_label_format = data_label_format.lower() if data_label_format else None
    if data_label_format is not None and data_label_format not in [
        "absolute",
        "percentage",
        "both",
    ]:
        raise ValueError(
            "data_label_format must be either 'absolute', 'percentage', 'both', or None.",
        )

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
    ax.yaxis.set_major_formatter(
        lambda x, pos: gu.human_format(x, pos, decimals=decimals),
    )

    # Add a black line at the y=0 position
    ax.axhline(y=0, color="black", linewidth=1, zorder=-1)

    if data_label_format is not None:
        labels = format_data_labels(
            df["amounts"],
            amount_total,
            data_label_format,
            decimals,
        )

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


def format_data_labels(
    amounts: pd.Series,
    total_change: float,
    label_format: str,
    decimals: int,
) -> list[str]:
    """Format the data labels based on the specified format."""
    if label_format == "absolute":
        return amounts.apply(lambda x: gu.human_format(x, decimals=decimals + 1))

    if label_format == "percentage":
        return amounts.apply(lambda x: f"{x / total_change:.0%}")

    return [f"{gu.human_format(x, decimals=decimals + 1)} ({x / total_change:.0%})" for x in amounts]
