"""This module provides functionality for creating bubble chart visualizations that display price distribution analysis across different categories.

The bubble chart shows price distribution as vertical layers (price bands) with bubble sizes representing the percentage of products in each price range for different categories like retailers, countries, etc.

### Core Features

- **Price Band Analysis**: Automatically bins price data into ranges using pandas.cut()
- **Categorical Grouping**: Groups data by categorical columns (retailers, countries, etc.)
- **Bubble Sizing**: Bubble sizes represent percentage of products in each price band per group
- **Flexible Binning**: Supports both integer (equal-width bins) and array (custom boundaries) inputs
- **Grid Layout**: X-axis shows categories, Y-axis shows price bands

### Use Cases

- **Retailer Price Comparison**: Compare price distributions across different retailers
- **Regional Price Analysis**: Analyze price positioning by country/region
- **Competitive Pricing**: Identify pricing gaps and opportunities
- **Price Architecture Visualization**: Visualize competitive pricing landscapes

### Limitations

- **Pandas DataFrame Only**: No Ibis table support
- **Pre-aggregated Data**: Data should be at product level (one row per product)
- **Numeric Price Column**: Requires numeric price/value column for binning
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib.lines import Line2D

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.plots.styles.tailwind import get_plot_colors


def _validate_inputs(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    bins: int | list[float],
) -> tuple[pd.DataFrame, int | list[float]]:
    """Validates and processes inputs for price distribution plotting.

    Args:
        df: Input DataFrame containing product-level data.
        value_col: Column containing the price/value data.
        group_col: Column containing the categorical grouping.
        bins: Either number of equal-width bins (int) or custom bin boundaries (list).

    Returns:
        Tuple of (cleaned_dataframe, validated_bins).

    Raises:
        ValueError: If DataFrame is empty, columns don't exist, value column is not numeric, or bins parameter is invalid.
        KeyError: If specified columns are not found in DataFrame.
        TypeError: If bins parameter has invalid type.
    """
    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")

    # Validate columns exist
    if value_col not in df.columns:
        msg = f"value_col '{value_col}' not found in DataFrame"
        raise KeyError(msg)

    if group_col not in df.columns:
        msg = f"group_col '{group_col}' not found in DataFrame"
        raise KeyError(msg)

    # Validate value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        msg = f"value_col '{value_col}' must be numeric for binning"
        raise ValueError(msg)

    # Validate bins parameter
    validated_bins = _validate_bins_parameter(bins)

    # Remove rows with missing values in key columns
    df_clean = df[[value_col, group_col]].dropna()

    if df_clean.empty:
        msg = f"No valid data after removing missing values from {value_col} and {group_col}"
        raise ValueError(msg)

    return df_clean, validated_bins


def _validate_bins_parameter(bins: int | list[float]) -> int | list[float]:
    """Validates and processes the bins parameter for price distribution plotting.

    Args:
        bins: Either number of equal-width bins (int) or custom bin boundaries (list).

    Returns:
        Validated and processed bins parameter.

    Raises:
        ValueError: If bins parameter is invalid.
        TypeError: If bins parameter has invalid type.
    """
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("bins must be a positive integer")
        return bins
    if isinstance(bins, list):
        min_bins = 2
        if len(bins) < min_bins:
            raise ValueError("bins list must contain at least 2 values")
        if not all(isinstance(x, int | float) for x in bins):
            raise ValueError("All values in bins list must be numeric")
        return sorted(bins)
    msg = "bins must be either an integer or a list of numeric values"
    raise TypeError(msg)


def plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    bins: int | list[float],
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Creates a bubble chart visualization showing price distribution analysis across categories.

    The chart displays price bands as vertical layers with bubble sizes representing the percentage
    of products in each price range for different groups (retailers, countries, etc.).

    Args:
        df (pd.DataFrame): Input DataFrame containing product-level data.
        value_col (str): Column containing the price/value data (e.g., "unit_price").
        group_col (str): Column containing the categorical grouping (e.g., "retailer").
        bins (int | list[float]): Either number of equal-width bins (int) or custom bin boundaries (list).
        title (str, optional): The title of the plot. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to None.
        y_label (str, optional): The label for the y-axis. Defaults to None.
        legend_title (str, optional): The title for the legend. Defaults to None.
        ax (Axes, optional): The Matplotlib Axes object to plot on. Defaults to None.
        source_text (str, optional): Text to be displayed as a source at the bottom of the plot. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area. Defaults to False.
        **kwargs (dict[str, Any]): Additional keyword arguments for the scatter plot function.

    Returns:
        SubplotBase: The Matplotlib Axes object with the generated bubble chart.

    Raises:
        ValueError: If DataFrame is empty, columns don't exist, or bins parameter is invalid.
        KeyError: If specified columns are not found in DataFrame.
        TypeError: If bins parameter has invalid type.
    """
    # Validate inputs and get clean data
    df_clean, bins = _validate_inputs(df, value_col, group_col, bins)

    # Create price bins
    df_clean["price_bin"] = pd.cut(df_clean[value_col], bins=bins, include_lowest=True)

    # Calculate percentage distribution for each group
    group_totals = df_clean.groupby(group_col, observed=True).size()
    bin_counts = df_clean.groupby([group_col, "price_bin"], observed=True).size().unstack(fill_value=0)

    # Convert to proportions (0-1 range)
    proportions = bin_counts.div(group_totals, axis=0)

    ax = ax or plt.gca()

    # Get unique groups and bins
    groups = proportions.index.tolist()
    price_bins = proportions.columns.tolist()

    # Set up color mapping
    colors = get_plot_colors(len(groups))

    alpha = kwargs.pop("alpha", 0.7)
    s_scale = kwargs.pop("s", 800)  # size for bubbles
    edge_color = kwargs.pop("edgecolor", "black")  # black stroke around bubbles
    line_width = kwargs.pop("linewidth", 1.5)  # Stroke width

    # Validate that we have some data
    if proportions.max().max() == 0 or pd.isna(proportions.max().max()):
        raise ValueError("All proportions are zero - no data falls within the specified bins")

    # Stack to get all (group, price_bin) combinations with their proportions
    stacked = proportions.stack()
    # Filter out zero proportions to avoid invisible bubbles
    stacked = stacked[stacked > 0]

    if len(stacked) > 0:  # Only plot if there are non-zero proportions
        x_positions = [groups.index(group) for group, _ in stacked.index]
        y_positions = [price_bins.index(price_bin) for _, price_bin in stacked.index]
        # Calculate bubble sizes using absolute proportion values for cross-group comparison
        bubble_sizes = []
        for group, price_bin in stacked.index:
            proportion_value = stacked.loc[(group, price_bin)]
            scaled_size = proportion_value * s_scale
            bubble_sizes.append(scaled_size)
        bubble_colors = [colors[groups.index(group)] for group, _ in stacked.index]

        ax.scatter(
            x_positions,
            y_positions,
            s=bubble_sizes,
            c=bubble_colors,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=line_width,
            **kwargs,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_yticks(range(len(price_bins)))

    # Format price bin labels to be more user-friendly
    formatted_labels = []
    for bin_ in price_bins:
        # Extract left and right bounds from pandas Interval
        left = bin_.left
        right = bin_.right
        # Format as price ranges
        formatted_labels.append(f"{left:.1f} - {right:.1f}")

    ax.set_yticklabels(formatted_labels)

    # Apply standard styling but completely disable legend handling
    ax = gu.standard_graph_styles(
        ax=ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_title=None,
        move_legend_outside=False,
        show_legend=False,
    )

    # Create custom legend with uniform circle sizes AFTER standard_graph_styles
    if len(groups) > 1:  # Only create legend if there are multiple groups
        legend_elements = []
        for i, _group in enumerate(groups):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[i],
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    alpha=0.8,
                    linestyle="None",
                ),
            )

        if move_legend_outside:
            ax.legend(
                legend_elements,
                groups,
                title=legend_title,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                frameon=False,
            )
        else:
            ax.legend(legend_elements, groups, title=legend_title, frameon=False)

    if source_text:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax=ax)
