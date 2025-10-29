"""This module provides flexible functionality for creating line plots from pandas DataFrames.

It focuses on visualizing sequences that are ordered or sequential but not necessarily categorical, such as "days since
an event" or "months since a competitor opened." However, while this module can handle datetime values on the x-axis,
the **plots.time_line** module has additional features that make working with datetimes easier, such as easily resampling the
data to alternate time frames.

The sequences used in this module can include values like "days since an event" (e.g., -2, -1, 0, 1, 2) or "months
since a competitor store opened." **This module is not intended for use with actual datetime values**.

### Core Features

- **Plotting Sequences or Indexes**: Plot one or more value columns (**`value_col`**) with support for sequences like
-2, -1, 0, 1, 2 (e.g., months since an event), using either the index or a specified x-axis column (**`x_col`**).
- **Custom X-Axis or Index**: Use any column as the x-axis (**`x_col`**) or plot based on the index if no x-axis column is specified.
- **Multiple Lines**: Create separate lines for each unique value in **`group_col`** (e.g., categories or product types).
- **Comprehensive Customization**: Easily customize plot titles, axis labels, and legends, with the option to move the legend outside the plot.
- **Pre-Aggregated Data**: The data must be pre-aggregated before plotting, as no aggregation occurs within the module.

### Use Cases

- **Daily Trends**: Plot trends such as daily revenue or user activity, for example, tracking revenue since the start of the year.
- **Event Impact**: Visualize how metrics (e.g., revenue, sales, or traffic) change before and after an important event, such as a competitor store opening or a product launch.
- **Category Comparison**: Compare metrics across multiple categories over time, for example, tracking total revenue for the top categories before and after an event like the introduction of a new competitor.

### Limitations and Handling of Temporal Data

- **Limited Handling of Temporal Data**: This module can plot simple time-based sequences, such as "days since an event," but it cannot manipulate or directly handle datetime or date-like columns. It is not optimized for actual datetime values.
If a datetime column is passed or more complex temporal plotting is needed, consider using the **`plots.time_line`** module, which is specifically designed for working with temporal data and performing time-based manipulation.
- **Pre-Aggregated Data Required**: The module does not perform any data aggregation, so all data must be pre-aggregated before being passed in for plotting.

"""

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.plots.styles.tailwind import COLORS, get_multi_color_cmap, get_single_color_cmap


def _validate_and_prepare_input(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None,
    x_col: str | None,
    group_col: str | None,
) -> tuple[pd.DataFrame, str | list[str]]:
    """Validate input parameters and convert Series to DataFrame if needed."""
    # Handle Series input
    if isinstance(df, pd.Series):
        if value_col is not None:
            raise ValueError(
                "When df is a pd.Series, value_col must be None. The Series itself represents the values to plot.",
            )
        if x_col is not None:
            raise ValueError(
                "When df is a pd.Series, x_col must be None. The Series index is used as the x-axis.",
            )
        if group_col is not None:
            raise ValueError(
                "When df is a pd.Series, group_col must be None. Cannot group a single series.",
            )
        # Convert Series to DataFrame for uniform processing
        series_name = df.name if df.name is not None else "value"
        df = df.to_frame(name=series_name)
        value_col = series_name

    # Validate value_col for DataFrame input
    if value_col is None:
        raise ValueError("value_col is required when df is a DataFrame")

    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")

    return df, value_col


def _validate_highlight_parameter(
    highlight: str | list[str] | None,
    value_col: str | list[str],
    group_col: str | None,
) -> list[str] | None:
    """Validate and normalize the highlight parameter."""
    if highlight is None:
        return None

    # Convert single string to list for uniform handling
    if isinstance(highlight, str):
        highlight = [highlight]

    # Check if this is a single-line plot
    is_single_line = group_col is None and (
        isinstance(value_col, str) or (isinstance(value_col, list) and len(value_col) == 1)
    )
    if is_single_line:
        raise ValueError("highlight parameter cannot be used with single-line plots")

    return highlight


def _create_pivot_dataframe(
    df: pd.DataFrame,
    value_col: str | list[str],
    x_col: str | None,
    group_col: str | None,
    fill_na_value: float | None,
) -> pd.DataFrame:
    """Create pivot DataFrame for plotting."""
    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = (
            df.pivot(columns=group_col, values=value_col)
            if x_col is None
            else df.pivot(index=x_col, columns=group_col, values=value_col)
        )
        if fill_na_value is not None:
            pivot_df = pivot_df.fillna(fill_na_value)

    return pivot_df


def _validate_highlight_values(
    highlight: list[str] | None,
    pivot_df: pd.DataFrame,
) -> list[str] | None:
    """Validate highlight values against available columns."""
    if highlight is None:
        return None

    available_columns = list(pivot_df.columns)
    invalid_highlights = [h for h in highlight if h not in available_columns]
    if invalid_highlights:
        error_msg = f"highlight values {invalid_highlights} not found in available columns {available_columns}"
        raise ValueError(error_msg)

    return highlight


def _categorize_columns(
    pivot_df: pd.DataFrame,
    highlight: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Categorize columns into highlighted and context groups."""
    if highlight is not None:
        highlighted_cols = [col for col in pivot_df.columns if col in highlight]
        context_cols = [col for col in pivot_df.columns if col not in highlight]
    else:
        highlighted_cols = list(pivot_df.columns)
        context_cols = []

    return highlighted_cols, context_cols


def _generate_colors(highlighted_cols: list[str]) -> list[str]:
    """Generate colors for highlighted lines."""
    color_gen_threshold = 4
    num_highlighted = len(highlighted_cols) if highlighted_cols else 1
    color_gen = get_single_color_cmap() if num_highlighted < color_gen_threshold else get_multi_color_cmap()
    return [next(color_gen) for _ in range(num_highlighted)]


def _render_plot(
    pivot_df: pd.DataFrame,
    highlighted_cols: list[str],
    context_cols: list[str],
    highlighted_colors: list[str],
    is_multi_line: bool,
    ax: Axes | None,
    **kwargs: dict[str, any],
) -> Axes:
    """Render the actual plot with context and highlighted lines."""
    # Context line styling
    context_color = COLORS["gray"][400]  # #9ca3af
    context_alpha = 0.6
    context_linewidth = 1.5

    # Highlighted line styling
    highlighted_linewidth = kwargs.pop("linewidth", 3)
    highlighted_alpha = 1.0

    # Plot context lines first (lower z-order)
    if context_cols:
        context_df = pivot_df[context_cols]
        ax = context_df.plot(
            ax=ax,
            linewidth=context_linewidth,
            color=context_color,
            alpha=context_alpha,
            legend=False,
            zorder=1,
            **{k: v for k, v in kwargs.items() if k not in ["color", "alpha", "zorder"]},
        )

    # Plot highlighted lines second (higher z-order)
    if highlighted_cols:
        highlighted_df = pivot_df[highlighted_cols]
        ax = highlighted_df.plot(
            ax=ax,
            linewidth=highlighted_linewidth,
            color=kwargs.pop("color", highlighted_colors),
            alpha=highlighted_alpha,
            legend=is_multi_line,
            zorder=2,
            **kwargs,
        )
    else:
        # Handle case where only context lines exist (shouldn't happen due to validation)
        ax = pivot_df.plot(
            ax=ax,
            linewidth=highlighted_linewidth,
            color=kwargs.pop("color", highlighted_colors),
            alpha=highlighted_alpha,
            legend=is_multi_line,
            **kwargs,
        )

    return ax


def _apply_final_styling(
    ax: Axes,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    legend_title: str | None,
    move_legend_outside: bool,
    source_text: str | None,
) -> SubplotBase:
    """Apply final styling and formatting to the plot."""
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


def plot(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    x_col: str | None = None,
    group_col: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    fill_na_value: float | None = None,
    highlight: str | list[str] | None = None,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots the `value_col` over the specified `x_col` or index, creating a separate line for each unique value in `group_col`.

    This function supports both pandas DataFrames and Series as input. When a Series is provided,
    the Series values are plotted against its index, and `value_col` must be None.

    Args:
        df (pd.DataFrame | pd.Series): The dataframe or series to plot. When a Series is provided,
            it represents the values to plot against its index.
        value_col (str, list of str, or None): The column(s) to plot. Must be None when df is a Series.
            Required when df is a DataFrame.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different lines.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        fill_na_value (float, optional): Value to fill NaNs with after pivoting.
        highlight (str or list of str, optional): Line(s) to highlight. When using
            `group_col`, these should be group values. When using a list of `value_col`,
            these should be column names. Highlighted lines will be rendered with bold
            linewidth (3), full opacity (alpha=1.0), and saturated colors. Non-highlighted
            lines will be muted with gray color (#9ca3af), reduced opacity (alpha=0.6),
            thinner linewidth (1.5), and rendered behind highlighted lines.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If `value_col` is a list and `group_col` is provided (which causes ambiguity in plotting).
        ValueError: If df is a Series and `value_col` is not None.
        ValueError: If df is a DataFrame and `value_col` is None.
        ValueError: If df is a Series and `x_col` is specified (Series uses its index as x-axis).
        ValueError: If df is a Series and `group_col` is specified (cannot group a single series).
        ValueError: If `highlight` is provided for single-line plot.
        ValueError: If `highlight` values don't match available groups/columns.

    Examples:
        Highlighting specific product categories:

        >>> import pandas as pd
        >>> from pyretailscience.plots import line
        >>> df = pd.DataFrame({
        ...     "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        ...     "category": ["Electronics", "Electronics", "Electronics",
        ...                  "Clothing", "Clothing", "Clothing",
        ...                  "Home", "Home", "Home"],
        ...     "revenue": [100, 120, 140, 80, 85, 90, 60, 65, 70]
        ... })
        >>> line.plot(
        ...     df=df,
        ...     x_col="month",
        ...     value_col="revenue",
        ...     group_col="category",
        ...     highlight=["Electronics", "Clothing"],  # Home will be muted
        ...     title="Revenue by Category (Electronics & Clothing Highlighted)"
        ... )

        Highlighting specific value columns:

        >>> df = pd.DataFrame({
        ...     "day": range(1, 6),
        ...     "revenue": [100, 110, 120, 130, 140],
        ...     "units_sold": [50, 55, 60, 65, 70],
        ...     "avg_order_value": [2.0, 2.0, 2.0, 2.0, 2.0],
        ...     "profit_margin": [0.2, 0.22, 0.24, 0.26, 0.28]
        ... })
        >>> line.plot(
        ...     df=df,
        ...     x_col="day",
        ...     value_col=["revenue", "units_sold", "avg_order_value", "profit_margin"],
        ...     highlight=["revenue", "profit_margin"],  # Other metrics muted
        ...     title="Daily Metrics (Revenue & Profit Margin Highlighted)"
        ... )

        Single highlighted line:

        >>> line.plot(
        ...     df=df,
        ...     x_col="month",
        ...     value_col="revenue",
        ...     group_col="category",
        ...     highlight="Electronics",  # str is acceptable for single highlight
        ...     title="Revenue by Category (Electronics Highlighted)"
        ... )
    """
    # Validate and prepare input
    df, value_col = _validate_and_prepare_input(df, value_col, x_col, group_col)

    # Validate highlight parameter
    highlight = _validate_highlight_parameter(highlight, value_col, group_col)

    # Create pivot DataFrame
    pivot_df = _create_pivot_dataframe(df, value_col, x_col, group_col, fill_na_value)

    # Validate highlight values
    highlight = _validate_highlight_values(highlight, pivot_df)

    # Categorize columns and generate colors
    highlighted_cols, context_cols = _categorize_columns(pivot_df, highlight)
    highlighted_colors = _generate_colors(highlighted_cols)

    # Determine if multi-line plot
    is_multi_line = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    # Render the plot
    ax = _render_plot(pivot_df, highlighted_cols, context_cols, highlighted_colors, is_multi_line, ax, **kwargs)

    # Apply final styling
    return _apply_final_styling(ax, title, x_label, y_label, legend_title, move_legend_outside, source_text)
