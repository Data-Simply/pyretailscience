"""This module provides functionality for creating scatter plots and bubble charts from pandas DataFrames.

It is designed to visualize relationships between variables, highlight distributions, and compare different categories using scatter points with optional variable sizing for bubble chart functionality.

### Core Features

- **Flexible X-Axis Handling**: Uses an index or a specified x-axis column (**`x_col`**) for plotting.
- **Multiple Scatter Groups**: Supports plotting multiple columns (**`value_col`**) or groups (**`group_col`**).
- **Bubble Chart Support**: Variable point sizes based on data values using **`size_col`** and **`size_scale`** parameters.
- **Point Labels**: Supports adding text labels to individual scatter points with automatic positioning to avoid overlaps.
- **Dynamic Color Mapping**: Automatically selects a colormap based on the number of groups.
- **Legend Customization**: Supports custom legend titles and the option to move the legend outside the plot.
- **Source Text**: Provides an option to add source attribution to the plot.

### Use Cases

- **Category-Based Scatter Plots**: Compare different categories using scatter points.
- **Bubble Charts**: Visualize three dimensions of data with x, y positions and point sizes representing a third variable.
- **Trend Analysis**: Identify patterns and outliers in datasets.
- **Multi-Value Scatter Plots**: Show multiple data series in a single scatter chart.
- **Labeled Scatter Plots**: Identify specific data points with text labels (e.g., product names, store IDs).
- **Store Performance Analysis**: Show sales vs profit with store size as bubble sizes.

### Bubble Chart Features

- **Variable Point Sizes**: Use `size_col` parameter to specify a column for point sizes.
- **Size Scaling**: Control bubble size scaling with `size_scale` parameter for optimal visualization.
- **Grouped Bubble Charts**: Combine bubble sizing with group-based coloring for multi-dimensional analysis.
- **Size Validation**: Automatic validation ensures size column contains numeric values.

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
- **Size Column Requirements**: When using bubble charts, `size_col` must contain numeric values only.
"""

import matplotlib.pyplot as plt
import pandas as pd
import textalloc as ta
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.plots.styles.styling_context import get_styling_context
from pyretailscience.plots.styles.tailwind import get_multi_color_cmap, get_single_color_cmap


def _validate_size_col(df: pd.DataFrame, size_col: str | None) -> None:
    """Validate size_col parameter.

    Args:
        df: DataFrame containing the data.
        size_col: Column name containing values to determine point sizes.

    Raises:
        KeyError: If size_col doesn't exist in DataFrame.
        ValueError: If DataFrame is empty or size_col contains non-numeric values.
    """
    if size_col is None:
        return

    if size_col not in df.columns:
        msg = f"size_col '{size_col}' not found in DataFrame"
        raise KeyError(msg)

    # Check for empty DataFrame first
    if len(df) == 0:
        msg = "Cannot create bubble chart with empty DataFrame"
        raise ValueError(msg)

    if not pd.api.types.is_numeric_dtype(df[size_col]):
        msg = f"size_col '{size_col}' must contain numeric values"
        raise ValueError(msg)


def _process_size_data(
    df: pd.DataFrame,
    size_col: str | None,
    size_scale: float,
    x_col: str | None,
    group_col: str | None,
) -> pd.DataFrame | pd.Series | None:
    """Process size data for bubble charts.

    Args:
        df: DataFrame containing the data.
        size_col: Column name containing values to determine point sizes.
        size_scale: Scaling factor for point sizes.
        x_col: Column name for x-values. If None, uses index.
        group_col: Column name for grouping. If None, treats as single series.

    Returns:
        Processed size data aligned with plot structure, or None if no size_col.
    """
    if size_col is None:
        return None

    if group_col is None:
        # For ungrouped data, align sizes with pivot_df index
        return df.set_index(x_col if x_col is not None else df.index)[size_col] * size_scale

    # For grouped data, create size array that aligns with pivot structure
    size_pivot = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=size_col)
    return size_pivot * size_scale


def _create_scatter_plot(
    ax: Axes,
    pivot_df: pd.DataFrame,
    colors: list,
    size_data: pd.DataFrame | pd.Series | None,
    group_col: str | None,
    is_multi_scatter: bool,
    alpha: float,
    **kwargs: dict[str, any],
) -> None:
    """Create scatter plots for each column in pivot_df.

    Args:
        ax: Matplotlib axes object to plot on.
        pivot_df: DataFrame with pivoted data for plotting.
        colors: List of colors for each column.
        size_data: Processed size data for bubble charts.
        group_col: Column name for grouping.
        is_multi_scatter: Whether this is a multi-series scatter plot.
        alpha: Alpha transparency value.
        **kwargs: Additional keyword arguments for matplotlib scatter function.
    """
    for col, color_val in zip(pivot_df.columns, colors, strict=False):
        # Get size values for this column if size_col is specified
        sizes = None
        if size_data is not None:
            sizes = size_data if group_col is None else size_data[col]

        # Filter out NaN values for grouped data
        if group_col is not None:
            # Get non-NaN mask for both y-values and sizes
            y_values = pivot_df[col]
            mask = y_values.notna()

            x_values = pivot_df.index[mask]
            y_values = y_values[mask]

            if sizes is not None:
                sizes = sizes[mask]
        else:
            x_values = pivot_df.index
            y_values = pivot_df[col]

        ax.scatter(
            x_values,
            y_values,
            s=sizes,
            color=color_val,
            label=col if is_multi_scatter else None,
            alpha=alpha,
            **kwargs,
        )


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
    size_col: str | None = None,
    size_scale: float = 1.0,
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
        size_col (str, optional): The column name containing values to determine point sizes.
            If None, all points have uniform size. Creates bubble charts when specified.
        size_scale (float, optional): Scaling factor for point sizes. Default: 1.0.
            Actual size = size_col_value * size_scale.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        legend_title (str, optional): The title of the legend.
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
        KeyError: If `size_col` doesn't exist in DataFrame.
        ValueError: If `size_col` contains non-numeric values.
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

    # Validate size_col parameter
    _validate_size_col(df, size_col)

    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)

    is_multi_scatter = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    color_gen_threshold = 3
    num_colors = len(pivot_df.columns) if is_multi_scatter else 1
    color_gen = get_single_color_cmap() if num_colors < color_gen_threshold else get_multi_color_cmap()
    default_colors = [next(color_gen) for _ in range(num_colors)]

    # Handle color parameter - can be single color or list of colors
    color = kwargs.pop("color", default_colors)
    colors = [color] * num_colors if not isinstance(color, list) else color

    # Process size data if size_col is specified
    size_data = _process_size_data(df, size_col, size_scale, x_col, group_col)

    ax = ax or plt.gca()
    alpha = kwargs.pop("alpha", 0.7)

    # Remove 's' from kwargs to avoid conflict with our size parameter
    kwargs.pop("s", None)
    _create_scatter_plot(ax, pivot_df, colors, size_data, group_col, is_multi_scatter, alpha, **kwargs)

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
