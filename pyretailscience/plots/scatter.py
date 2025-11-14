"""This module provides functionality for creating scatter plots from pandas DataFrames.

It is designed to visualize relationships between variables, highlight distributions, and compare different categories using scatter points.

### Core Features

- **Flexible X-Axis Handling**: Uses an index or a specified x-axis column (**`x_col`**) for plotting.
- **Multiple Scatter Groups**: Supports plotting multiple columns (**`value_col`**) or groups (**`group_col`**).
- **Point Labels**: Supports adding text labels to individual scatter points with automatic positioning to avoid overlaps.
- **Dynamic Color Mapping**: Automatically selects a colormap based on the number of groups.
- **Legend Customization**: Supports custom legend titles and the option to move the legend outside the plot.
- **Source Text**: Provides an option to add source attribution to the plot.

### Use Cases

- **Category-Based Scatter Plots**: Compare different categories using scatter points.
- **Trend Analysis**: Identify patterns and outliers in datasets.
- **Multi-Value Scatter Plots**: Show multiple data series in a single scatter chart.
- **Labeled Scatter Plots**: Identify specific data points with text labels (e.g., product names, store IDs).

### Label Support

- **Single Series Labeling**: When using a single `value_col`, labels can be added via `label_col` parameter.
- **Group-Based Labeling**: When using `group_col`, each point gets labeled from the original DataFrame.
- **Automatic Label Positioning**: Uses textalloc library to prevent label overlaps and optimize readability.
- **Clean Label Display**: Labels are positioned without connecting lines to maintain a clean appearance.
- **Customizable Label Styling**: Control label appearance through `label_kwargs` parameter.

### Limitations and Warnings

- **Pre-Aggregated Data Required**: The module does not perform data aggregation; data should be pre-aggregated
before being passed to the function.
- **Label Limitations**: Point labels are not supported when `value_col` is a list (raises ValueError).
"""

import matplotlib.pyplot as plt
import pandas as pd
import textalloc as ta
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.plots.styles.styling_context import get_styling_context
from pyretailscience.plots.styles.tailwind import get_plot_colors


def _add_point_labels(
    ax: Axes,
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    x_col: str | None = None,
    group_col: str | None = None,
    label_kwargs: dict[str, any] | None = None,
) -> None:
    """Add text labels to scatter plot points with automatic positioning.

    Args:
        ax: Matplotlib axes object to add labels to.
        df: DataFrame containing the data.
        value_col: Column name for y-values.
        label_col: Column name containing text labels.
        x_col: Column name for x-values. If None, uses index.
        group_col: Column name for grouping. If None, treats as single series.
        label_kwargs: Additional arguments passed to textalloc.allocate().
    """
    # Get styling context for font properties
    styling_context = get_styling_context()

    # Prepare data for labeling - drop rows with NaN in value_col, label_col, or x_col
    cols_to_check = [value_col, label_col]
    if x_col is not None:
        cols_to_check.append(x_col)
    data_with_labels_df = df.dropna(subset=cols_to_check)

    # Get x-values (either from x_col or use index)
    x_values = data_with_labels_df[x_col] if x_col is not None else data_with_labels_df.index

    if group_col is None:
        # Single series - vectorized approach
        all_x_coords = x_values.tolist()
        all_y_coords = data_with_labels_df[value_col].tolist()
        all_labels = data_with_labels_df[label_col].astype(str).tolist()
    else:
        # Multi-series with groups - vectorized approach
        all_x_coords = data_with_labels_df.index.to_numpy() if x_col is None else data_with_labels_df[x_col].to_numpy()
        all_y_coords = data_with_labels_df[value_col].to_numpy()
        all_labels = data_with_labels_df[label_col].astype(str).to_numpy()

    # Apply textalloc to avoid overlaps
    if len(all_x_coords) > 0 and len(all_y_coords) > 0 and len(all_labels) > 0:
        # Set default textalloc parameters
        allocate_kwargs = {
            "textsize": styling_context.fonts.data_label_size,
            "x_scatter": all_x_coords,
            "y_scatter": all_y_coords,
            "nbr_candidates": 50,  # More candidates for better positioning
            "draw_lines": False,  # Remove lines to scatter points
        }

        # Override with user-provided kwargs
        if label_kwargs:
            allocate_kwargs.update(label_kwargs)

        ta.allocate(
            ax,
            all_x_coords,
            all_y_coords,
            all_labels,
            **allocate_kwargs,
        )


def plot(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_col: str | None = None,
    group_col: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    label_col: str | None = None,
    label_kwargs: dict[str, any] | None = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots a scatter chart for the given `value_col` over `x_col` or index, with optional grouping by `group_col`.

    Args:
        df (pd.DataFrame or pd.Series): The dataframe or series to plot.
        value_col (str or list of str): The column(s) to plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different scatter groups.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        label_col (str, optional): Column name containing text labels for each point.
            Not supported when value_col is a list. Defaults to None.
        label_kwargs (dict, optional): Keyword arguments passed to textalloc.allocate().
            Common options: textsize, nbr_candidates, min_distance, max_distance, draw_lines.
            By default, draw_lines=False to avoid lines connecting labels to points.
            Defaults to None.
        **kwargs: Additional keyword arguments for matplotlib scatter function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If `value_col` is a list and `group_col` is provided (which causes ambiguity in plotting).
        ValueError: If `label_col` is provided when `value_col` is a list.
        KeyError: If `label_col` doesn't exist in DataFrame.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")

    if label_col is not None:
        if isinstance(value_col, list):
            raise ValueError(
                "label_col is not supported when value_col is a list. "
                "Please use a single value_col or create separate plots.",
            )

        if label_col not in df.columns:
            msg = f"label_col '{label_col}' not found in DataFrame"
            raise KeyError(msg)

    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)

    is_multi_scatter = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    num_colors = len(pivot_df.columns) if is_multi_scatter else 1
    default_colors = get_plot_colors(num_colors)

    # Handle color parameter - can be single color or list of colors
    color = kwargs.pop("color", default_colors)
    colors = [color] * num_colors if not isinstance(color, list) else color

    ax = ax or plt.gca()
    alpha = kwargs.pop("alpha", 0.7)
    for col, color_val in zip(pivot_df.columns, colors, strict=False):
        ax.scatter(
            pivot_df.index,
            pivot_df[col],
            color=color_val,
            label=col if is_multi_scatter else None,
            alpha=alpha,
            **kwargs,
        )

    # Add labels if requested
    if label_col is not None:
        _add_point_labels(
            ax=ax,
            df=df,
            value_col=value_col,
            label_col=label_col,
            x_col=x_col,
            group_col=group_col,
            label_kwargs=label_kwargs,
        )

    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
    )

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax)
