"""This module provides versatile plotting functionality for creating line plots from pandas DataFrames.

It is designed for plotting sequences that resemble time-based data, such as "days" or "months" since
an event, but it does not explicitly handle datetime values. For actual time-based plots (using datetime
objects), please refer to the `time_plot` module.

The sequences used in this module can include values such as "days since an event" (e.g., -2, -1, 0, 1, 2)
or "months since a competitor store opened." **This module is not intended for use with actual datetime values**.
If a datetime or datetime-like column is passed as `x_col`, a warning will be triggered suggesting the use
of the `time_plot` module.

Key Features:
--------------
- **Plotting Sequences or Indexes**: Plot one or more value columns (`value_col`), supporting sequences
  like -2, -1, 0, 1, 2 (e.g., months since an event), using either the index or a specified x-axis
  column (`x_col`).
- **Custom X-Axis or Index**: Use any column as the x-axis (`x_col`), or plot based on the index if no
  x-axis column is specified.
- **Multiple Lines**: Create separate lines for each unique value in `group_col` (e.g., categories).
- **Comprehensive Customization**: Easily customize titles, axis labels, legends, and optionally move
  the legend outside the plot.
- **Pre-Aggregated Data**: The data must be pre-aggregated before plotting. No aggregation occurs in
  this module.

### Common Scenarios and Examples:

1. **Basic Plot Showing Daily Revenue Trends Since The Start of the Current Year**:

    This example demonstrates how to plot the `total_price` on a daily basis since the start of the year.

    **Preparing the Data**:
    ```python
    import numpy as np
    import pandas as pd

    # Convert 'transaction_datetime' to a datetime column if it's not already
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])

    # Create a column for days since the start of the year
    df['days_since_start_of_year'] = (df['transaction_datetime'] - pd.Timestamp(f'{df["transaction_datetime"].dt.year[0]}-01-01')).dt.days

    # Aggregate data by days since the start of the year
    df_daily = df.groupby('days_since_start_of_year').agg({'total_price': 'sum'}).reset_index()

    ```

    **Plotting**:
    ```python
    ax = line.plot(
        df=df_daily,
        value_col="total_price",  # Plot 'total_price' values
        x_col="days_since_start_of_year",  # Use 'days_since_start_of_year' as the x-axis
        title="Daily Revenue Since Start of Year",  # Title of the plot
        x_label="Days Since Start of Year",  # X-axis label
        y_label="Daily Revenue",  # Y-axis label
    )

    plt.show()
    ```

    **Use Case**: This is useful when you want to illustrate revenue trends throughout the year on a daily basis. The
    x-axis represents days since the last year  and the y-axis shows how revenue behaved over this period.

---


2. **Basic Plot Showing Price Trends Since Competitor Store Opened**:

    This example demonstrates how to plot the `total_price` over the number of months since a competitor
    store opened. The total revenue remains stable or increases slightly before the store opened, and then
    drops randomly after the competitor's store opened.

    **Preparing the Data**:
    ```python
    import numpy as np

    # Convert 'transaction_datetime' to a datetime column if it's not already
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])

    # Resample the data by month
    df['month'] = df['transaction_datetime'].dt.to_period('M')  # Extract year and month
    df_monthly = df.groupby('month').agg({'total_price': 'sum'}).reset_index()

    # Create the "months since competitor opened" column
    # Assume the competitor opened 60% of the way through the data
    competitor_opened_month_index = int(len(df_monthly) * 0.6)
    df_monthly['months_since_competitor_opened'] = np.arange(-competitor_opened_month_index, len(df_monthly) - competitor_opened_month_index)

    # Simulate stable or increasing revenue before competitor opened
    df_monthly.loc[df_monthly['months_since_competitor_opened'] < 0, 'total_price'] *= np.random.uniform(1.05, 1.2)

    # Simulate a random drop after the competitor opened
    df_monthly.loc[df_monthly['months_since_competitor_opened'] >= 0, 'total_price'] *= np.random.uniform(0.8, 0.95, size=len(df_monthly[df_monthly['months_since_competitor_opened'] >= 0]))
    ```

    **Plotting**:
    ```python
    ax = line.plot(
        df=df_monthly,
        value_col="total_price",  # Plot 'total_price' values
        x_col="months_since_competitor_opened",  # Use 'months_since_competitor_opened' as the x-axis
        title="Total Revenue Since Competitor Store Opened",  # Title of the plot
        x_label="Months Since Competitor Opened",  # X-axis label
        y_label="Total Revenue",  # Y-axis label
    )

    plt.show()
    ```

    **Use Case**: This is useful when you want to illustrate the effect of a competitor store opening
    on sales performance. The x-axis represents months before and after the event, and the y-axis shows
    how revenue behaved over time.

---

3. **Plotting Price Trends by Category (Top 3 Categories)**:

    This example plots the total revenue for the top 3 categories before and after the competitor opened.
    The data is resampled by month, split by category, and tracks the months since the competitor store opened.

    **Preparing the Data**:
    ```python
    import numpy as np
    import pandas as pd

    # Convert 'transaction_datetime' to a datetime column if it's not already
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])

    # Resample the data by month and category
    df['month'] = df['transaction_datetime'].dt.to_period('M')  # Extract year and month
    df_monthly = df.groupby(['month', 'category_0_name']).agg({'total_price': 'sum'}).reset_index()

    # Create a separate dataframe for unique months to track "months since competitor opened"
    unique_months = df_monthly['month'].unique()
    competitor_opened_month_index = int(len(unique_months) * 0.6)  # Assume competitor opened 60% of the way through

    # Create 'months_since_competitor_opened' for each unique month
    months_since_competitor_opened = np.concatenate([
        np.arange(-competitor_opened_month_index, 0),  # Before competitor opened
        np.arange(0, len(unique_months) - competitor_opened_month_index)  # After competitor opened
    ])

    # Create a new dataframe with the 'months_since_competitor_opened' values and merge it back
    months_df = pd.DataFrame({'month': unique_months, 'months_since_competitor_opened': months_since_competitor_opened})
    df_monthly = df_monthly.merge(months_df, on='month', how='left')

    # Filter to include months both before and after the competitor opened
    df_since_competitor_opened = df_monthly[(df_monthly['months_since_competitor_opened'] >= -6) &  # Include 6 months before
                                            (df_monthly['months_since_competitor_opened'] <= 12)]  # Include 12 months after

    # Identify top 3 categories based on total_price across the selected period
    category_totals = df_since_competitor_opened.groupby('category_0_name')['total_price'].sum().sort_values(ascending=False)

    # Get the top 3 categories
    top_categories = category_totals.head(3).index

    # Filter the dataframe to include only the top 3 categories
    df_top_categories = df_since_competitor_opened[df_since_competitor_opened['category_0_name'].isin(top_categories)]
    ```

    **Plotting**:
    ```python
    ax = line.plot(
        df=df_top_categories,
        value_col="total_price",  # Plot 'total_price' values
        group_col="category_0_name",  # Separate lines for each category
        x_col="months_since_competitor_opened",  # Use 'months_since_competitor_opened' as the x-axis
        title="Total Revenue for Top 3 Categories (Before and After Competitor Opened)",  # Title of the plot
        x_label="Months Since Competitor Opened",  # X-axis label
        y_label="Total Revenue",  # Y-axis label
        legend_title="Category"  # Legend title
    )

    plt.show()
    ```

    **Use Case**: Use this when you want to analyze the behavior of specific top categories before and after
    an event, such as the opening of a competitor store.

---

### Customization Options:
- **`value_col`**: The column or list of columns to plot (e.g., `'total_price'`).
- **`group_col`**: A column whose unique values will be used to create separate lines (e.g., `'category_0_name'`).
- **`x_col`**: The column to use as the x-axis (e.g., `'months_since_competitor_opened'`). **Warning**: If a datetime
  or datetime-like column is passed, a warning will suggest using the `time_plot` module instead.
- **`title`**, **`x_label`**, **`y_label`**: Custom text for the plot title and axis labels.
- **`legend_title`**: Custom title for the legend based on `group_col`.
- **`move_legend_outside`**: Boolean flag to move the legend outside the plot.

---

### Dependencies:
- `pandas`: For DataFrame manipulation and grouping.
- `matplotlib`: For generating plots.
- `pyretailscience.style.graph_utils`: For applying consistent graph styles across the plots.

"""
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.style.tailwind import get_base_cmap

if TYPE_CHECKING:
    from matplotlib.colors import ListedColormap


def _check_datetime_column(df: pd.DataFrame, x_col: str) -> None:
    """Checks if the x_col is a datetime or convertible to datetime.

    Issues a warning if the column is datetime-like, recommending
    the use of a time-based plot.

    Args:
        df (pd.DataFrame): The dataframe containing the column to check.
        x_col (str): The column to check for datetime-like values.
    """
    if x_col not in df.columns:
        msg = f"The column '{x_col}' is not present in the dataframe."
        raise KeyError(msg)

    try:
        pd.to_datetime(df[x_col], errors="raise")
        warnings.warn(
            f"The column '{x_col}' can be converted to datetime. Consider using the 'time_plot' module for time-based "
            f"plots.",
            UserWarning,
            stacklevel=2,
        )

    except (ValueError, TypeError):
        return


def plot(
    df: pd.DataFrame,
    value_col: str | list[str],
    x_label: str,
    y_label: str,
    title: str,
    x_col: str | None = None,
    group_col: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    **kwargs: dict[str, any],
) -> SubplotBase:
    """Plots the `value_col` over the specified `x_col` or index, creating a separate line for each unique value in `group_col`.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str or list of str): The column(s) to plot.
        x_label (str): The x-axis label.
        y_label (str): The y-axis label.
        title (str): The title of the plot.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different lines.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        SubplotBase: The matplotlib axes object.
    """
    if x_col is not None:
        _check_datetime_column(df, x_col)

    colors: ListedColormap = get_base_cmap()

    if group_col is not None:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)
    else:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[value_col]

    ax = pivot_df.plot(
        ax=ax,
        linewidth=3,
        color=colors.colors[: len(pivot_df.columns) if group_col else 1],
        legend=(group_col is not None),
        **kwargs,
    )

    ax = gu.standard_graph_styles(
        ax,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    if move_legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1))

    if legend_title is not None:
        ax.legend(title=legend_title)

    if source_text is not None:
        gu.add_source_text(ax=ax, source_text=source_text)

    return gu.standard_tick_styles(ax)
