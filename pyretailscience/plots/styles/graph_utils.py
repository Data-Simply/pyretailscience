"""Helper functions for styling graphs."""

import importlib.resources as pkg_resources
from collections.abc import Generator
from datetime import datetime
from itertools import cycle
from typing import Literal

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.dates import date2num
from matplotlib.text import Text
from scipy import stats

from pyretailscience.plots.styles.styling_helpers import PlotStyler

ASSETS_PATH = pkg_resources.files("pyretailscience").joinpath("assets")
_MAGNITUDE_SUFFIXES = ["", "K", "M", "B", "T", "P"]


def _hatches_gen() -> Generator[str, None, None]:
    """Returns a generator that cycles through predefined hatch patterns.

    Yields:
        str: The next hatch pattern in the sequence.
    """
    _hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    return cycle(_hatches)


def human_format(
    num: float,
    pos: int | None = None,  # noqa: ARG001 (pos is only used for Matplotlib compatibility)
    decimals: int = 0,
    prefix: str = "",
) -> str:
    """Format a number in a human-readable format for Matplotlib, discarding trailing zeros.

    Args:
        num (float): The number to format.
        pos (int, optional): The position. Defaults to None. Only used for Matplotlib compatibility.
        decimals (int, optional): The number of decimals. Defaults to 0.
        prefix (str, optional): The prefix of the returned string, eg '$'. Defaults to "".

    Returns:
        str: The formatted number, with trailing zeros removed.
    """
    # The minimum difference between two numbers to receive a different suffix
    minimum_magnitude_difference = 1000.0
    magnitude = 0

    # Keep dividing by 1000 until the number is small enough
    while abs(num) >= minimum_magnitude_difference:
        magnitude += 1
        num /= minimum_magnitude_difference

    # Check if the number rounds to exactly 1000 at the current magnitude
    if round(abs(num), decimals) == minimum_magnitude_difference:
        num /= minimum_magnitude_difference
        magnitude += 1

    # If magnitude exceeds the predefined suffixes, continue with multiples of "P"
    if magnitude < len(_MAGNITUDE_SUFFIXES):
        suffix = _MAGNITUDE_SUFFIXES[magnitude]
    else:
        # Calculate how many times beyond "P" we've gone and append that to "P"
        extra_magnitude = magnitude - (len(_MAGNITUDE_SUFFIXES) - 1)
        suffix = f"{1000**extra_magnitude}P"

    # Format the number and remove trailing zeros
    formatted_num = f"{prefix}%.{decimals}f" % num
    formatted_num = formatted_num.rstrip("0").rstrip(".") if "." in formatted_num else formatted_num

    return f"{formatted_num}{suffix}"


def truncate_to_x_digits(num_str: str, digits: int) -> str:
    """Truncate a human-formatted number to the first `num_digits` significant digits.

    Args:
        num_str (str): The formatted number (e.g., '999.999K').
        digits (int): The number of digits to keep.

    Returns:
        str: The truncated formatted number (e.g., '999.9K').
    """
    # Split the number part and the suffix (e.g., "999.999K" -> "999.999" and "K")
    suffix = ""
    for s in _MAGNITUDE_SUFFIXES:
        if num_str.endswith(s) and s != "":
            suffix = s
            num_str = num_str[: -len(s)]  # Remove the suffix for now
            break

    # Handle negative numbers
    is_negative = num_str.startswith("-")
    if is_negative:
        num_str = num_str[1:]  # Remove the negative sign for now

    # Handle zero case explicitly
    if float(num_str) == 0:
        return f"0{suffix}"

    # Handle small numbers explicitly to avoid scientific notation
    scientific_notation_threshold = 1e-4
    if abs(float(num_str)) < scientific_notation_threshold:
        return f"{float(num_str):.{digits}f}".rstrip("0").rstrip(".")

    digits_before_decimal = len(num_str.split(".")[0])
    # Calculate how many digits to keep after the decimal
    digits_to_keep_after_decimal = digits - digits_before_decimal

    # Ensure truncation without rounding
    if digits_to_keep_after_decimal > 0:
        factor = 10**digits_to_keep_after_decimal
        truncated_num = str(int(float(num_str) * factor) / factor)
    else:
        factor = 10**digits
        truncated_num = str(int(float(num_str) * factor) / factor)

    # Reapply the negative sign if needed
    if is_negative:
        truncated_num = f"-{truncated_num}"

    # Remove unnecessary trailing zeros and decimal point
    truncated_num = truncated_num.rstrip("0").rstrip(".")

    return f"{truncated_num}{suffix}"


def standard_graph_styles(
    ax: Axes,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title_pad: int | None = None,
    x_label_pad: int | None = None,
    y_label_pad: int | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    show_legend: bool = True,
) -> Axes:
    """Apply standard styles to a Matplotlib graph using styling helpers.

    Args:
        ax (Axes): The graph to apply the styles to.
        title (str, optional): The title of the graph. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None.
        y_label (str, optional): The y-axis label. Defaults to None.
        title_pad (int, optional): The padding above the title. Defaults to styling context default.
        x_label_pad (int, optional): The padding below the x-axis label. Defaults to styling context default.
        y_label_pad (int, optional): The padding to the left of the y-axis label. Defaults to styling context default.
        legend_title (str, optional): The title of the legend. If None, no legend title is applied. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot. Defaults to False.
        show_legend (bool): Whether to display the legend or not.

    Returns:
        Axes: The graph with the styles applied.
    """
    plot_styler = PlotStyler()

    # Apply base plot styling
    plot_styler.apply_base_styling(ax)

    # Apply text styling
    if title is not None:
        plot_styler.apply_title(ax, title, title_pad)

    if x_label is not None:
        plot_styler.apply_label(ax, x_label, "x", x_label_pad)

    if y_label is not None:
        plot_styler.apply_label(ax, y_label, "y", y_label_pad)

    # Apply tick styling
    plot_styler.apply_ticks(ax)

    # Apply legend styling if needed
    if show_legend and (ax.get_legend() is not None or legend_title is not None or move_legend_outside):
        plot_styler.apply_legend(ax, legend_title, move_legend_outside)

    return ax


def standard_tick_styles(ax: Axes) -> Axes:
    """Apply standard tick styles using styling helpers.

    Args:
        ax (Axes): The graph to apply the styles to.

    Returns:
        Axes: The graph with the styles applied.
    """
    plot_styler = PlotStyler()
    plot_styler.apply_ticks(ax)
    return ax


def apply_hatches(ax: Axes, num_segments: int) -> Axes:
    """Apply hatch patterns to patches in a plot, such as bars, histograms, or area plots.

    This function divides the patches in the given Axes object into the specified
    number of segments and applies a different hatch pattern to each segment.

    Args:
        ax (Axes): The matplotlib Axes object containing the plot with patches (bars, histograms, etc.).
        num_segments (int): The number of segments to divide the patches into, with each segment receiving a different hatch pattern.

    Returns:
        Axes: The modified Axes object with hatches applied to the patches.
    """
    available_hatches = _hatches_gen()
    patch_groups = np.array_split(ax.patches, num_segments)
    for patch_group in patch_groups:
        hatch = next(available_hatches)
        for patch in patch_group:
            patch.set_hatch(hatch)

    legend = ax.get_legend()
    if legend:
        existing_hatches = [patch.get_hatch() for patch in ax.patches if patch.get_hatch() is not None]
        unique_hatches = [hatch for idx, hatch in enumerate(existing_hatches) if hatch not in existing_hatches[:idx]]
        for legend_patch, hatch in zip(legend.get_patches(), cycle(unique_hatches)):
            legend_patch.set_hatch(hatch)

    return ax


def not_none(value1: any, value2: any) -> any:
    """Helper function that returns the first value that is not None.

    Args:
        value1: The first value.
        value2: The second value.

    Returns:
        The first value that is not None.
    """
    if value1 is None:
        return value2
    return value1


def get_decimals(ylim: tuple[float, float], tick_values: list[float], max_decimals: int = 10) -> int:
    """Helper function for the `human_format` function that determines the number of decimals to use for the y-axis.

    Args:
        ylim: The y-axis limits.
        tick_values: The y-axis tick values.
        max_decimals: The maximum number of decimals to use. Defaults to 100.

    Returns:
        int: The number of decimals to use.
    """
    decimals = 0
    while decimals < max_decimals:
        tick_labels = [human_format(t, 0, decimals=decimals) for t in tick_values if t >= ylim[0] and t <= ylim[1]]
        # Ensure no duplicate labels
        if len(tick_labels) == len(set(tick_labels)):
            break
        decimals += 1
    return decimals


def add_source_text(
    ax: Axes,
    source_text: str,
    font_size: float | None = None,
    vertical_padding: float = 2,
    is_venn_diagram: bool = False,
) -> Text:
    """Add source text to the bottom left corner of a graph using styling helpers.

    Args:
        ax (Axes): The graph to add the source text to.
        source_text (str): The source text.
        font_size (float, optional): The font size of the source text. If None, uses styling context default.
        vertical_padding (float, optional): The padding in ems below the x-axis label. Defaults to 2.
        is_venn_diagram (bool, optional): Flag to indicate if the diagram is a Venn diagram.
            If True, `x_norm` and `y_norm` will be set to fixed values. Defaults to False.

    Returns:
        Text: The source text.
    """
    plot_styler = PlotStyler()

    return plot_styler.apply_source_text(
        ax=ax,
        text=source_text,
        font_size=font_size,
        vertical_padding=vertical_padding,
        is_venn_diagram=is_venn_diagram,
    )


def set_axis_percent(
    fmt_axis: YAxis | XAxis,
    xmax: float = 1,
    decimals: int | None = None,
    symbol: str | None = "%",
) -> None:
    """Format an axis to display values as percentages.

    This function configures a matplotlib axis to display its tick labels as percentages
    using matplotlib's PercentFormatter.

    Args:
        fmt_axis (YAxis | XAxis): The axis to format (either ax.yaxis or ax.xaxis).
        xmax (float, optional): The value that represents 100%. Defaults to 1.
        decimals (int | None, optional): Number of decimal places to include. If None,
            automatically selects based on data range. Defaults to None.
        symbol (str | None, optional): The symbol to use for percentage. If None,
            no symbol is displayed. Defaults to "%".

    Returns:
        None: The function modifies the axis formatter in place.

    Example:
        ```python
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 0.25, 0.5, 0.75, 1.0], [0, 0.3, 0.5, 0.7, 1.0])
        # Format y-axis as percentage
        set_axis_percent(ax.yaxis)
        # Format x-axis as percentage with 1 decimal place
        set_axis_percent(ax.xaxis, decimals=1)
        ```
    """
    return fmt_axis.set_major_formatter(mtick.PercentFormatter(xmax=xmax, decimals=decimals, symbol=symbol))


def _calculate_r_squared_original_space(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate R² in original data space.

    Args:
        y_actual (np.ndarray): Actual y values.
        y_predicted (np.ndarray): Predicted y values from regression model.

    Returns:
        float: R² value calculated in original data space.
    """
    ss_res = np.sum((y_actual - y_predicted) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)  # Total sum of squares

    # Handle edge case where all y values are identical
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


def _perform_regression_calculation(
    regression_type: str, x_filtered: np.ndarray, y_filtered: np.ndarray
) -> tuple[float, float, float]:
    """Perform regression calculation and return coefficients and R² in original data space.

    Args:
        regression_type (str): Type of regression to perform.
        x_filtered (np.ndarray): Filtered x data.
        y_filtered (np.ndarray): Filtered y data.

    Returns:
        tuple[float, float, float]: param1, param2, r_squared (calculated in original data space)
    """
    if regression_type == "linear":
        # Linear regression: y = mx + b
        slope, intercept, r_value, _, _ = stats.linregress(x_filtered, y_filtered)
        return slope, intercept, r_value**2

    if regression_type == "power":
        # Power law regression: y = ax^b → log(y) = log(a) + b*log(x)
        log_x = np.log(x_filtered)
        log_y = np.log(y_filtered)
        slope, intercept, _, _, _ = stats.linregress(log_x, log_y)
        a = np.exp(intercept)  # Convert back: a = exp(intercept), b = slope
        b = slope

        # Calculate R² in original data space
        y_predicted = a * (x_filtered**b)
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    if regression_type == "logarithmic":
        # Logarithmic regression: y = a*ln(x) + b
        log_x = np.log(x_filtered)
        slope, intercept, _, _, _ = stats.linregress(log_x, y_filtered)
        a = slope  # a = slope, b = intercept
        b = intercept

        # Calculate R² in original data space
        y_predicted = a * np.log(x_filtered) + b
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    if regression_type == "exponential":
        # Exponential regression: y = ae^(bx) → ln(y) = ln(a) + bx
        log_y = np.log(y_filtered)
        slope, intercept, _, _, _ = stats.linregress(x_filtered, log_y)
        a = np.exp(intercept)  # Convert back: a = exp(intercept), b = slope
        b = slope

        # Calculate R² in original data space
        y_predicted = a * np.exp(b * x_filtered)
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    msg = f"Unsupported regression type: {regression_type}"
    raise ValueError(msg)


def _generate_regression_line(
    regression_type: str, param1: float, param2: float, x_min: float, x_max: float, data_size: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Generate regression line points for plotting with adaptive point calculation.

    Args:
        regression_type (str): Type of regression.
        param1 (float): First parameter (slope/a coefficient).
        param2 (float): Second parameter (intercept/b coefficient).
        x_min (float): Minimum x value for line.
        x_max (float): Maximum x value for line.
        data_size (int): Number of original data points for adaptive calculation.

    Returns:
        tuple[np.ndarray, np.ndarray]: x_line, y_line arrays for plotting.
    """
    if regression_type == "linear":
        # Linear: use endpoints for efficiency
        x_line = np.array([x_min, x_max])
        y_line = param1 * x_line + param2
        return x_line, y_line

    # For non-linear types, use adaptive point calculation
    # Base points on data size but ensure smooth curves for complex functions
    min_points = 50
    max_points = 500
    adaptive_points = max(data_size * 3, min_points)
    num_points = min(adaptive_points, max_points)

    if regression_type == "power":
        # Ensure we don't go below a small positive value to avoid log issues
        x_start = max(x_min, 1e-6)
        x_line = np.linspace(x_start, x_max, num_points)
        y_line = param1 * (x_line**param2)
        return x_line, y_line

    if regression_type == "logarithmic":
        # Ensure we don't go below a small positive value to avoid log issues
        x_start = max(x_min, 1e-6)
        x_line = np.linspace(x_start, x_max, num_points)
        y_line = param1 * np.log(x_line) + param2
        return x_line, y_line

    if regression_type == "exponential":
        x_line = np.linspace(x_min, x_max, num_points)
        y_line = param1 * np.exp(param2 * x_line)
        return x_line, y_line

    msg = f"Unsupported regression type: {regression_type}"
    raise ValueError(msg)


def _add_equation_text(
    ax: Axes,
    param1: float,
    param2: float,
    r_squared: float,
    color: str,
    text_position: float,
    show_equation: bool,
    show_r2: bool,
    regression_type: str = "linear",
) -> None:
    """Add equation and R² text to the plot.

    Args:
        ax (Axes): The matplotlib axes object.
        param1 (float): First regression parameter (slope/a coefficient).
        param2 (float): Second regression parameter (intercept/b coefficient).
        r_squared (float): The R² value of the regression.
        color (str): The color of the text.
        text_position (float): The relative y-position of the text.
        show_equation (bool): Whether to display the equation.
        show_r2 (bool): Whether to display the R² value.
        regression_type (str): The type of regression for equation formatting.
    """
    if not (show_equation or show_r2):
        return

    plot_styler = PlotStyler()

    equation_parts = []

    if show_equation:
        if regression_type == "linear":
            sign = "+" if param2 >= 0 else "-"
            equation = f"y = {param1:.3f}x {sign} {abs(param2):.3f}"
        elif regression_type == "power":
            equation = f"y = {param1:.3f}x^{param2:.3f}"
        elif regression_type == "logarithmic":
            sign = "+" if param2 >= 0 else "-"
            equation = f"y = {param1:.3f}ln(x) {sign} {abs(param2):.3f}"
        elif regression_type == "exponential":
            equation = f"y = {param1:.3f}e^({param2:.3f}x)"
        else:
            equation = f"y = {param1:.3f}x + {param2:.3f}"  # Fallback

        equation_parts.append(equation)

    if show_r2:
        r2_text = f"R² = {r_squared:.3f}"
        equation_parts.append(r2_text)

    text = "\n".join(equation_parts)

    # Calculate text position (relative to axis bounds)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    text_x = x_min + 0.05 * (x_max - x_min)  # 5% from left
    text_y = y_min + text_position * (y_max - y_min)

    ax.text(
        text_x,
        text_y,
        text,
        color=color,
        fontsize=plot_styler.context.fonts.label_size,
        fontproperties=plot_styler.context.get_font_properties(plot_styler.context.fonts.source_font),
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


def _extract_plot_data(ax: Axes) -> tuple[np.ndarray, np.ndarray]:
    """Extract x and y data from a matplotlib plot (line, scatter, or bar).

    Supports multiple plot types:
    - Line plots: Extracts data from line objects
    - Bar charts: Extracts center positions and heights/widths, with automatic orientation detection
    - Scatter plots: Extracts data from collection offsets

    Args:
        ax (Axes): The matplotlib axes object containing the plot.

    Returns:
        tuple[np.ndarray, np.ndarray]: The x and y data arrays.

    Raises:
        ValueError: If no plot data can be extracted.
    """
    # Try to get data from lines first (line plots)
    lines = [line for line in ax.get_lines() if line.get_visible()]

    if len(lines) > 0:
        x_data = lines[0].get_xdata()
        y_data = lines[0].get_ydata()
    # Check for bar charts (patches)
    elif hasattr(ax, "patches") and ax.patches:
        # Detect bar orientation using BarContainer (stable API)
        is_vertical = True  # Default assumption
        for container in ax.containers:
            if hasattr(container, "orientation") and container.orientation:
                is_vertical = container.orientation == "vertical"
                break

        if is_vertical:
            # Vertical bars: x is center position, y is height
            bar_data = [(patch.get_x() + patch.get_width() / 2, patch.get_height()) for patch in ax.patches]
        else:
            # Horizontal bars: x is width, y is center position
            bar_data = [(patch.get_width(), patch.get_y() + patch.get_height() / 2) for patch in ax.patches]

        # Sort by x-coordinate to ensure consistency with grouped/stacked bar charts
        bar_data.sort(key=lambda point: point[0])
        x_data, y_data = np.array(bar_data).T
    # If no lines or bars, check for scatter plots (or other collections)
    elif hasattr(ax, "collections") and ax.collections:
        # Extract data from the first collection (e.g., scatter plot)
        collection = ax.collections[0]
        # Get the offsets which contain the x,y coordinates
        if hasattr(collection, "get_offsets") and callable(collection.get_offsets):
            offset_data = collection.get_offsets()
            if len(offset_data) > 0:
                x_data = offset_data[:, 0]
                y_data = offset_data[:, 1]
            else:
                raise ValueError("No data points found in the collection.")
        else:
            raise ValueError("Cannot extract data from this type of collection.")
    else:
        raise ValueError("No visible lines or collections found in the plot.")

    return x_data, y_data


def _prepare_numeric_data(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert plot data to numeric arrays suitable for regression analysis.

    Args:
        x_data (np.ndarray): The raw x-axis data from the plot.
        y_data (np.ndarray): The raw y-axis data from the plot.

    Returns:
        tuple[np.ndarray, np.ndarray]: The numeric x and y data.

    Raises:
        ValueError: If data cannot be converted to numeric format or has insufficient valid points.
    """
    # Simple fallback indices in case we can't process the data
    x_indices = np.arange(len(x_data))

    # Check if x_data contains datetime objects
    is_datetime = False
    try:
        # Try to find a non-null value to check its type
        for val in x_data:
            if val is not None:
                is_datetime = isinstance(val, datetime | pd.Timestamp)
                break
    except (TypeError, IndexError):
        pass

    try:
        # Handle datetime or numeric data appropriately
        x_numeric = date2num(x_data) if is_datetime else np.array(x_data, dtype=float)
    except (TypeError, ValueError):
        # Fallback to simple indices if conversion fails
        x_numeric = x_indices

    try:
        y_numeric = np.array(y_data, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("Cannot convert y-axis values to numeric format for regression") from err

    # Create mask to filter out NaN values
    valid_mask = ~np.isnan(x_numeric) & ~np.isnan(y_numeric)
    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points for regression")

    # Check that we have enough valid data points for regression
    min_points_for_regression = 2
    if np.sum(valid_mask) < min_points_for_regression:
        error_msg = f"At least {min_points_for_regression} valid data points are required for regression analysis"
        raise ValueError(error_msg)

    return x_numeric[valid_mask], y_numeric[valid_mask]


def _validate_regression_data(
    x_data: np.ndarray, y_data: np.ndarray, regression_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and filter data for specific regression types.

    Args:
        x_data (np.ndarray): The x-axis data.
        y_data (np.ndarray): The y-axis data.
        regression_type (str): The regression type being used.

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered x and y data arrays.

    Raises:
        ValueError: If insufficient valid data remains after filtering.
    """
    if regression_type == "power":
        # Power regression requires positive x AND y values for log transformation
        positive_mask = (x_data > 0) & (y_data > 0)
        min_regression_points = 2
        if np.sum(positive_mask) < min_regression_points:
            raise ValueError("Power regression requires at least 2 data points with positive x and y values")
        x_filtered = x_data[positive_mask]
        y_filtered = y_data[positive_mask]
        return x_filtered, y_filtered

    if regression_type == "logarithmic":
        # Logarithmic requires positive x values for log transformation
        positive_x_mask = x_data > 0
        if np.sum(positive_x_mask) < min_regression_points:
            raise ValueError("Logarithmic regression requires at least 2 positive x values")
        x_filtered = x_data[positive_x_mask]
        y_filtered = y_data[positive_x_mask]
        return x_filtered, y_filtered

    if regression_type == "exponential":
        # Exponential requires positive y values for log transformation
        positive_y_mask = y_data > 0
        if np.sum(positive_y_mask) < min_regression_points:
            raise ValueError("Exponential regression requires at least 2 positive y values")
        x_filtered = x_data[positive_y_mask]
        y_filtered = y_data[positive_y_mask]
        return x_filtered, y_filtered

    # Linear regression uses all data
    return x_data, y_data


def add_regression_line(
    ax: Axes,
    regression_type: Literal["linear", "power", "logarithmic", "exponential"] = "linear",
    color: str = "red",
    linestyle: str = "--",
    text_position: float = 0.6,
    show_equation: bool = True,
    show_r2: bool = True,
    **kwargs: dict[str, any],
) -> Axes:
    """Add a regression line with configurable algorithm to a matplotlib plot.

    This function examines the data in a matplotlib Axes object and adds a
    regression line to it. It supports line plots, scatter plots, and bar charts
    (both vertical and horizontal), and can handle both numeric and datetime x-axis values.

    For bar charts, the function automatically detects orientation using matplotlib's
    BarContainer API and extracts appropriate x,y coordinates from bar positions and heights.

    Note: If an axes contains multiple plot types (e.g., both lines and bars), the function
    processes them in priority order: lines first, then bars, then scatter plots. Only the
    first available plot type will be used for regression analysis.

    Args:
        ax (Axes): The matplotlib axes object containing the plot (line, scatter, or bar).
        regression_type (Literal[...], optional): Regression algorithm to use.
            - "linear": y = mx + b (default, OLS regression)
            - "power": y = ax^b (elasticity analysis, log-log transformation)
            - "logarithmic": y = a*ln(x) + b (diminishing returns analysis)
            - "exponential": y = ae^(bx) (growth/decay patterns)
            Defaults to "linear".
        color (str, optional): Color of the regression line. Defaults to "red".
        linestyle (str, optional): Style of the regression line. Defaults to "--".
        text_position (float, optional): Relative position (0-1) for the equation text. Defaults to 0.6.
        show_equation (bool, optional): Whether to display the equation on the plot. Defaults to True.
        show_r2 (bool, optional): Whether to display the R² value on the plot. Defaults to True.
        kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        Axes: The matplotlib axes with the regression line added.

    Raises:
        ValueError: If the plot contains no visible lines, scatter points, or bar patches, or if
            regression_type is not supported.

    Examples:
        Basic linear regression (backward compatible):
        >>> ax = data.plot.scatter(x='price', y='demand')
        >>> gu.add_regression_line(ax)

        Power law regression for price elasticity:
        >>> gu.add_regression_line(ax, regression_type="power")

        Bar chart with regression line:
        >>> ax = df.plot.bar(x='category', y='sales')
        >>> gu.add_regression_line(ax, regression_type="linear")
    """
    # Validate regression type
    supported_types = ["linear", "power", "logarithmic", "exponential"]
    if regression_type not in supported_types:
        error_msg = f"Unsupported regression_type '{regression_type}'. Supported types: {supported_types}"
        raise ValueError(error_msg)

    # Extract data from the plot
    x_data, y_data = _extract_plot_data(ax)

    # Convert to numeric data and validate
    x_numeric, y_numeric = _prepare_numeric_data(x_data, y_data)

    # Apply algorithm-specific data validation and filtering
    x_filtered, y_filtered = _validate_regression_data(x_numeric, y_numeric, regression_type)

    # Perform regression calculation
    param1, param2, r_squared = _perform_regression_calculation(regression_type, x_filtered, y_filtered)

    # Generate regression line points
    x_min, x_max = ax.get_xlim()
    data_size = len(x_filtered)
    x_line, y_line = _generate_regression_line(regression_type, param1, param2, x_min, x_max, data_size)

    # Plot the regression line
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, **kwargs)

    # Add equation and R² text
    _add_equation_text(ax, param1, param2, r_squared, color, text_position, show_equation, show_r2, regression_type)

    return ax
