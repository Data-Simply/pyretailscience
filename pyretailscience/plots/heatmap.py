"""This module provides functionality for creating generic heatmap plots from pandas DataFrames.

This module is designed to create flexible heatmap visualizations suitable for various use cases
including migration matrices, confusion matrices, correlation matrices, and other 2D data
visualizations. It provides a clean, reusable interface without domain-specific assumptions.

### Core Features

- **Generic Design**: No domain-specific assumptions or hardcoded elements
- **Color Mapping**: Uses Tailwind green colormap for consistent visualization
- **Auto-contrast Text**: Text color automatically switches between black and white based on cell intensity
- **Customizable Labels**: Supports custom labels for x-axis, y-axis, title, and colorbar
- **Grid Styling**: White grid lines between cells for clear separation
- **Flexible Data**: Displays values as-is without formatting assumptions

### Use Cases

- **Migration Matrices**: Visualize customer movement between segments
- **Correlation Matrices**: Show relationships between variables
- **Confusion Matrices**: Display classification results
- **Any 2D Data**: Generic support for any tabular data visualization

### Design Principles

- Display values as-is from the DataFrame (no percentage or other formatting assumptions)
- Consistent with existing PyRetailScience plotting modules (line.py, bar.py)
- Minimal parameters with **kwargs for advanced customization
- Match visual style of existing plots while remaining generic
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.plots.styles.tailwind import get_listed_cmap

_LABEL_ROTATION_THRESHOLD = 10


def plot(
    df: pd.DataFrame,
    cbar_label: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    figsize: tuple[int, int] | None = None,
    cbar_format: str = "{x:.2f}",
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Creates a generic heatmap visualization from a pandas DataFrame.

    This function creates a color-coded heatmap with cell values displayed as text. It is suitable
    for visualizing any 2D data structure including migration matrices, confusion matrices,
    correlation matrices, or cohort analysis data.

    Args:
        df (pd.DataFrame): DataFrame to visualize. Index becomes y-axis, columns become x-axis.
        cbar_label (str): Label for the colorbar.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): Additional source text annotation.
        figsize (tuple[int, int], optional): The size of the plot. Defaults to None.
        cbar_format (str, optional): Format string for colorbar values. Defaults to "{x:.2f}".
        **kwargs: Additional keyword arguments passed to matplotlib's imshow function.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    cmap = get_listed_cmap("green")
    im = ax.imshow(df, cmap=cmap, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, format=cbar_format)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize="x-large")

    # Set up ticks and labels
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))

    # Handle long labels with rotation and proper alignment
    x_labels = df.columns.astype(str).to_list()
    y_labels = df.index.astype(str).to_list()

    # Determine if we need rotation based on label length
    max_x_label_length = max(len(label) for label in x_labels) if x_labels else 0
    rotation_angle = 45 if max_x_label_length > _LABEL_ROTATION_THRESHOLD else 0

    ax.set_xticklabels(x_labels, rotation=rotation_angle, ha="right" if rotation_angle > 0 else "center")
    ax.set_yticklabels(y_labels)

    # Position x-axis labels on bottom with extra padding for rotated labels
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Add extra padding at bottom if labels are rotated
    if rotation_angle > 0:
        ax.tick_params(axis="x", which="major", pad=10)

    # Create grid lines between cells
    ax.set_xticks(np.arange(df.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(df.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Calculate threshold for auto-contrast text
    threshold = im.norm(df.to_numpy().max()) / 2.0
    textcolors = ("black", "white")

    # Add text to each cell with auto-contrast
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            color = textcolors[int(im.norm(value) > threshold)]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7)

    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )
    ax.grid(False)

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax)
