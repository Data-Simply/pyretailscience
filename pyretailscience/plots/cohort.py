"""This module provides functionality for creating cohort plots from pandas DataFrames.

It is designed to visualize data distributions using color-coded heatmaps, helping to highlight trends and comparisons between different groups.

### Core Features

- **Color Mapping**: Uses a predefined colormap for visualizing data.
- **Customizable Labels**: Supports custom labels for x-axis, y-axis, title, and colorbar.
- **Source Text**: Provides an option to add source attribution to the plot.
- **Grid and Tick Customization**: Applies standard styling for better readability.

### Use Cases

- **Cohort Analysis**: Visualize how different groups behave over time.
- **Category-Based Heatmaps**: Compare values across different categories.

### Default Behavior

- **Percentage Display**: By default, values are displayed as percentages (e.g., "50%").
  Set `percentage=False` for raw number display (e.g., "0.50").

### Limitations and Warnings

- **Data Aggregation Required**: The module does not perform data aggregation; data should be pre-aggregated before being passed to the function.
- **Fixed Color Mapping**: The module uses a predefined colormap without dynamic adjustments.
"""

import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes, SubplotBase

from pyretailscience.plots import heatmap


def plot(
    df: pd.DataFrame,
    cbar_label: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    percentage: bool = True,
    figsize: tuple[int, int] | None = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots a cohort plot for the given DataFrame.

    Args:
        df (pd.DataFrame): Dataframe containing cohort analysis data.
        cbar_label (str): Label for the colorbar.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): Additional source text annotation.
        percentage (bool, optional): If True, displays cohort values as percentages. Defaults to True.
        figsize (tuple[int, int], optional): The size of the plot. Defaults to None.
        **kwargs: Additional keyword arguments for cohort styling.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    # Use generic heatmap
    ax = heatmap.plot(
        df=df,
        cbar_label=cbar_label,
        x_label=x_label,
        y_label=y_label,
        title=title,
        ax=ax,
        source_text=source_text,
        figsize=figsize,
        **kwargs,
    )

    # Add cohort-specific styling
    if percentage:
        # 1. Update colorbar format for percentages
        cbar = ax.figure.axes[-1]  # Get the colorbar axes
        cbar.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        # 2. Update cell text to show percentages
        for text in ax.texts:
            # Re-format text as percentage
            value = float(text.get_text())
            text.set_text(f"{value:.0%}")

    # 3. Add cohort-specific horizontal line
    ax.hlines(y=3 - 0.5, xmin=-0.5, xmax=df.shape[1] - 0.5, color="white", linewidth=4)

    return ax
