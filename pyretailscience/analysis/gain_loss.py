"""This module performs gain loss analysis (switching analysis) on a DataFrame to assess customer movement between brands or products over time.

Gain loss analysis, also known as switching analysis, is a marketing analytics technique used to
assess customer movement between brands or products over time. It helps businesses understand the dynamics of customer
acquisition and churn. Here's a concise definition: Gain loss analysis examines the flow of customers to and from a
brand or product, quantifying:

1. Gains: New customers acquired from competitors
2. Losses: Existing customers lost to competitors
3. Net change: The overall impact on market share

This analysis helps marketers:

- Identify trends in customer behavior
- Evaluate the effectiveness of marketing strategies
- Understand competitive dynamics in the market

The GainLoss class supports both pandas DataFrames and Ibis tables for high-performance analysis:

Example with pandas DataFrame (backward compatible):
    ```python
    import pandas as pd
    from pyretailscience.analysis.gain_loss import GainLoss

    # Traditional usage with pandas
    gl = GainLoss(
        df=transactions_df,  # Auto-converted to ibis.memtable
        p1_index=transactions_df['date'].between('2024-01-01', '2024-01-31'),
        p2_index=transactions_df['date'].between('2024-02-01', '2024-02-29'),
        focus_group_index=transactions_df['brand'] == 'Brand A',
        focus_group_name='Brand A',
        comparison_group_index=transactions_df['brand'] == 'Brand B',
        comparison_group_name='Brand B',
    )

    # Results available via lazy evaluation
    result = gl.df  # Computed on first access, cached
    ```

Example with Ibis tables (new capability):
    ```python
    import ibis
    from pyretailscience.analysis.gain_loss import GainLoss

    # Connect to database
    conn = ibis.duckdb.connect('retail.db')
    transactions_table = conn.table('transactions')

    gl = GainLoss(
        df=transactions_table,  # Direct Ibis table
        p1_index=transactions_table['date'].between('2024-01-01', '2024-01-31').to_pandas(),
        p2_index=transactions_table['date'].between('2024-02-01', '2024-02-29').to_pandas(),
        focus_group_index=(transactions_table['brand'] == 'Brand A').to_pandas(),
        focus_group_name='Brand A',
        comparison_group_index=(transactions_table['brand'] == 'Brand B').to_pandas(),
        comparison_group_name='Brand B',
    )

    # Inspect the generated SQL query
    print(gl.table.compile())

    # Query executed in database, not pandas
    result = gl.df
    ```
"""

import ibis
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.plots import bar
from pyretailscience.plots.styles.tailwind import COLORS


class GainLoss:
    """A class to perform gain loss analysis on a DataFrame to assess customer movement between brands or products over time."""

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        p1_index: list[bool] | pd.Series,
        p2_index: list[bool] | pd.Series,
        focus_group_index: list[bool] | pd.Series,
        focus_group_name: str,
        comparison_group_index: list[bool] | pd.Series,
        comparison_group_name: str,
        group_col: str | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> None:
        """Calculate the gain loss table for a given DataFrame at the customer level.

        Args:
            df (pd.DataFrame | ibis.Table): The DataFrame or Ibis table to calculate the gain loss table from.
            p1_index (list[bool]): The index for the first time period.
            p2_index (list[bool]): The index for the second time period.
            focus_group_index (list[bool]): The index for the focus group.
            focus_group_name (str): The name of the focus group.
            comparison_group_index (list[bool]): The index for the comparison group.
            comparison_group_name (str): The name of the comparison group.
            group_col (str | None, optional): The column to group by. Defaults to None.
            value_col (str, optional): The column to calculate the gain loss from. Defaults to option column.unit_spend.
            agg_func (str, optional): The aggregation function to use. Defaults to "sum".
        """
        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)
        elif not isinstance(df, ibis.Table):
            raise TypeError("df must be either a pandas DataFrame or an ibis Table")

        # Validate index lengths
        if not len(p1_index) == len(p2_index) == len(focus_group_index) == len(comparison_group_index):
            raise ValueError(
                "p1_index, p2_index, focus_group_index, and comparison_group_index should have the same length",
            )

        # Validate no overlap between time periods
        if any(p1 and p2 for p1, p2 in zip(p1_index, p2_index, strict=False)):
            raise ValueError("p1_index and p2_index should not overlap")

        # Validate no overlap between focus and comparison groups
        if any(focus and comp for focus, comp in zip(focus_group_index, comparison_group_index, strict=False)):
            raise ValueError("focus_group_index and comparison_group_index should not overlap")

        required_cols = [get_option("column.customer_id"), value_col] + ([group_col] if group_col is not None else [])
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.focus_group_name = focus_group_name
        self.comparison_group_name = comparison_group_name
        self.group_col = group_col
        self.value_col = value_col

        # Calculate the gain loss table using Ibis
        gain_loss_table_ibis = self._calc_gain_loss(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            comparison_group_index=comparison_group_index,
            group_col=group_col,
            value_col=value_col,
            agg_func=agg_func,
        )
        # Store the final Ibis table
        self.table = self._calc_gains_loss_table(
            gain_loss_table=gain_loss_table_ibis,
            group_col=group_col,
        )

        # Store intermediate Ibis table for backward compatibility lazy properties
        self._gain_loss_table_ibis = gain_loss_table_ibis

    @property
    def df(self) -> pd.DataFrame:
        """Returns the gain/loss analysis as a pandas DataFrame.

        Lazily evaluates the Ibis expression on first access.

        Returns:
            pd.DataFrame: The gain/loss analysis results
        """
        if self._df is None:
            self._df = self.table.execute()
        return self._df

    @property
    def gain_loss_df(self) -> pd.DataFrame:
        """Returns the customer-level gain/loss analysis as a pandas DataFrame.

        Backward compatibility property. Lazily evaluates the Ibis expression on first access.

        Returns:
            pd.DataFrame: The customer-level gain/loss analysis results
        """
        if not hasattr(self, "_gain_loss_df"):
            self._gain_loss_df = self._gain_loss_table_ibis.execute()
        return self._gain_loss_df

    @staticmethod
    def process_customer_group(
        focus_p1: float,
        comparison_p1: float,
        focus_p2: float,
        comparison_p2: float,
        focus_diff: float,
        comparison_diff: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Process the gain loss for a customer group.

        Note:
            This method is kept for backward compatibility, testing, and as a reference
            implementation of the business logic. The main Ibis implementation uses
            vectorized expressions in _apply_business_logic_ibis() for performance.
            This method is primarily used in test cases via parametrization to verify
            the core business logic implementation.

        Args:
            focus_p1 (float | int): The focus group total in the first time period.
            comparison_p1 (float | int): The comparison group total in the first time period.
            focus_p2 (float | int): The focus group total in the second time period.
            comparison_p2 (float | int): The comparison group total in the second time period.
            focus_diff (float | int): The difference in the focus group totals.
            comparison_diff (float | int): The difference in the comparison group totals.

        Returns:
            tuple[float, float, float, float, float, float]: The gain loss for the customer group.
        """
        if focus_p1 == 0 and comparison_p1 == 0:
            return focus_p2, 0, 0, 0, 0, 0
        if focus_p2 == 0 and comparison_p2 == 0:
            return 0, -1 * focus_p1, 0, 0, 0, 0

        if focus_diff > 0:
            focus_inc_dec = focus_diff if comparison_diff > 0 else max(0, comparison_diff + focus_diff)
        elif comparison_diff < 0:
            focus_inc_dec = focus_diff
        else:
            focus_inc_dec = min(0, comparison_diff + focus_diff)

        increased_focus = max(0, focus_inc_dec)
        decreased_focus = min(0, focus_inc_dec)

        transfer = focus_diff - focus_inc_dec
        switch_from_comparison = max(0, transfer)
        switch_to_comparison = min(0, transfer)

        return 0, 0, increased_focus, decreased_focus, switch_from_comparison, switch_to_comparison

    @staticmethod
    def _calc_gain_loss(
        df: ibis.Table,
        p1_index: list[bool],
        p2_index: list[bool],
        focus_group_index: list[bool],
        comparison_group_index: list[bool],
        group_col: str | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> ibis.Table:
        """Calculate the gain loss table for a given DataFrame at the customer level.

        Args:
            df (ibis.Table): The Ibis table to calculate the gain loss table from.
            p1_index (list[bool]): The index for the first time period.
            p2_index (list[bool]): The index for the second time period.
            focus_group_index (list[bool]): The index for the focus group.
            comparison_group_index (list[bool]): The index for the comparison group.
            group_col (str | None, optional): The column to group by. Defaults to None.
            value_col (str, optional): The column to calculate the gain loss from. Defaults to option column.unit_spend.
            agg_func (str, optional): The aggregation function to use. Defaults to "sum".

        Returns:
            ibis.Table: The gain loss table.
        """
        cols = ColumnHelper()

        # Convert boolean indices to integer indices for filtering (optimized with numpy)
        p1_indices = np.where(p1_index)[0].tolist()
        p2_indices = np.where(p2_index)[0].tolist()
        focus_indices = np.where(focus_group_index)[0].tolist()
        comparison_indices = np.where(comparison_group_index)[0].tolist()

        # Add row numbers to enable filtering
        df = df.mutate(_row_num=ibis.row_number())

        # Create boolean columns for filtering
        df = df.mutate(
            _is_p1=df._row_num.isin(p1_indices),
            _is_p2=df._row_num.isin(p2_indices),
            _is_focus=df._row_num.isin(focus_indices),
            _is_comparison=df._row_num.isin(comparison_indices),
        )

        # Filter to only rows in either period
        df_filtered = df.filter(df._is_p1 | df._is_p2)

        # Determine grouping columns
        grp_cols = [cols.customer_id] if group_col is None else [group_col, cols.customer_id]

        # Define aggregation function
        agg_func_attr = getattr(df_filtered[value_col], agg_func, None)
        if agg_func_attr is None:
            msg = f"Aggregation function '{agg_func}' not supported"
            raise ValueError(msg)

        # Create all aggregations in a single operation to avoid join issues
        result = df_filtered.group_by(grp_cols).aggregate(
            focus_p1=ibis.cases((df_filtered._is_focus & df_filtered._is_p1, df_filtered[value_col]), else_=0).sum(),
            comparison_p1=ibis.cases(
                (df_filtered._is_comparison & df_filtered._is_p1, df_filtered[value_col]),
                else_=0,
            ).sum(),
            total_p1=ibis.cases(
                ((df_filtered._is_focus | df_filtered._is_comparison) & df_filtered._is_p1, df_filtered[value_col]),
                else_=0,
            ).sum(),
            focus_p2=ibis.cases((df_filtered._is_focus & df_filtered._is_p2, df_filtered[value_col]), else_=0).sum(),
            comparison_p2=ibis.cases(
                (df_filtered._is_comparison & df_filtered._is_p2, df_filtered[value_col]),
                else_=0,
            ).sum(),
            total_p2=ibis.cases(
                ((df_filtered._is_focus | df_filtered._is_comparison) & df_filtered._is_p2, df_filtered[value_col]),
                else_=0,
            ).sum(),
        )

        # Calculate differences
        result = result.mutate(
            focus_diff=result.focus_p2 - result.focus_p1,
            comparison_diff=result.comparison_p2 - result.comparison_p1,
            total_diff=result.total_p2 - result.total_p1,
        )

        # Apply business logic using Ibis case expressions
        return GainLoss._apply_business_logic_ibis(result)

    @staticmethod
    def _apply_business_logic_ibis(table: ibis.Table) -> ibis.Table:
        """Apply the process_customer_group business logic using Ibis case expressions.

        Args:
            table: Ibis table with focus_p1, comparison_p1, focus_p2, comparison_p2, focus_diff, comparison_diff columns

        Returns:
            ibis.Table: Table with new, lost, increased_focus, decreased_focus, switch_from_comparison, switch_to_comparison columns
        """
        # Handle edge cases first: new customers (no p1 activity) and lost customers (no p2 activity)
        new_customers = (table.focus_p1 == 0) & (table.comparison_p1 == 0)
        lost_customers = (table.focus_p2 == 0) & (table.comparison_p2 == 0)

        # For new customers, all focus_p2 is "new"
        new = ibis.cases((new_customers, table.focus_p2), else_=0)

        # For lost customers, all focus_p1 is "lost" (negative)
        lost = ibis.cases((lost_customers, -1 * table.focus_p1), else_=0)

        # For existing customers (not new or lost), apply the complex logic
        existing_customers = ~new_customers & ~lost_customers

        # Calculate focus_inc_dec following the original logic
        focus_inc_dec = ibis.cases(
            (existing_customers & (table.focus_diff > 0) & (table.comparison_diff > 0), table.focus_diff),
            (existing_customers & (table.focus_diff > 0), ibis.greatest(0, table.comparison_diff + table.focus_diff)),
            (existing_customers & (table.comparison_diff < 0), table.focus_diff),
            (existing_customers, ibis.least(0, table.comparison_diff + table.focus_diff)),
            else_=0,
        )

        # Split focus_inc_dec into positive (increased) and negative (decreased) components
        increased_focus = ibis.greatest(0, focus_inc_dec)
        decreased_focus = ibis.least(0, focus_inc_dec)

        # Calculate transfer amounts (switching between groups)
        transfer = table.focus_diff - focus_inc_dec
        switch_from_comparison = ibis.greatest(0, transfer)
        switch_to_comparison = ibis.least(0, transfer)

        # Add all the calculated columns to the table
        return table.mutate(
            new=new,
            lost=lost,
            increased_focus=increased_focus,
            decreased_focus=decreased_focus,
            switch_from_comparison=switch_from_comparison,
            switch_to_comparison=switch_to_comparison,
        )

    @staticmethod
    def _calc_gains_loss_table(
        gain_loss_table: ibis.Table,
        group_col: str | None = None,
    ) -> ibis.Table:
        """Aggregates the gain loss table to show the total gains and losses across customers.

        Args:
            gain_loss_table (ibis.Table): The gain loss table at customer level to aggregate.
            group_col (str | None, optional): The column to group by. Defaults to None.

        Returns:
            ibis.Table: The aggregated gain loss table
        """
        # Define columns to aggregate
        agg_cols = [
            "focus_p1",
            "comparison_p1",
            "total_p1",
            "focus_p2",
            "comparison_p2",
            "total_p2",
            "focus_diff",
            "comparison_diff",
            "total_diff",
            "new",
            "lost",
            "increased_focus",
            "decreased_focus",
            "switch_from_comparison",
            "switch_to_comparison",
        ]

        # Create aggregation dictionary
        aggs = {col: gain_loss_table[col].sum() for col in agg_cols if col in gain_loss_table.columns}

        if group_col is None:
            # Aggregate across all rows
            return gain_loss_table.aggregate(**aggs)
        # Group by the specified column and aggregate
        return gain_loss_table.group_by(group_col).aggregate(**aggs)

    def plot(
        self,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        source_text: str | None = None,
        move_legend_outside: bool = False,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the gain loss table using the bar.plot wrapper.

        Args:
            title (str | None, optional): The title of the plot. Defaults to None.
            x_label (str | None, optional): The x-axis label. Defaults to None.
            y_label (str | None, optional): The y-axis label. Defaults to None.
            ax (Axes | None, optional): The axes to plot on. Defaults to None.
            source_text (str | None, optional): The source text to add to the plot. Defaults to None.
            move_legend_outside (bool, optional): Whether to move the legend outside the plot. Defaults to False.
            kwargs (dict[str, any]): Additional keyword arguments to pass to the plot.

        Returns:
            SubplotBase: The plot
        """
        green_colors = [COLORS["green"][700], COLORS["green"][500], COLORS["green"][300]]
        red_colors = [COLORS["red"][700], COLORS["red"][500], COLORS["red"][300]]

        increase_cols = ["new", "increased_focus", "switch_from_comparison"]
        decrease_cols = ["lost", "decreased_focus", "switch_to_comparison"]
        all_cols = increase_cols + decrease_cols

        plot_df = self.df.copy()
        default_y_label = self.focus_group_name if self.group_col is None else self.group_col
        plot_data = plot_df.copy()

        color_dict = {col: green_colors[i] for i, col in enumerate(increase_cols)}
        color_dict.update({col: red_colors[i] for i, col in enumerate(decrease_cols)})

        kwargs.pop("stacked", None)

        ax = bar.plot(
            df=plot_data,
            value_col=all_cols,
            title=gu.not_none(title, f"Gain Loss from {self.focus_group_name} to {self.comparison_group_name}"),
            y_label=gu.not_none(y_label, default_y_label),
            x_label=gu.not_none(x_label, self.value_col),
            orientation="horizontal",
            ax=ax,
            source_text=source_text,
            move_legend_outside=move_legend_outside,
            stacked=True,
            **kwargs,
        )

        for i, container in enumerate(ax.containers):
            col_name = all_cols[i]
            for patch in container:
                patch.set_color(color_dict[col_name])

        legend_labels = [
            "New",
            f"Increased {self.focus_group_name}",
            f"Switch From {self.comparison_group_name}",
            "Lost",
            f"Decreased {self.focus_group_name}",
            f"Switch To {self.comparison_group_name}",
        ]

        if ax.get_legend():
            ax.get_legend().remove()

        legend = ax.legend(
            legend_labels,
            frameon=True,
            bbox_to_anchor=(1.05, 1) if move_legend_outside else None,
            loc="upper left" if move_legend_outside else "best",
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

        ax.axvline(0, color="black", linewidth=0.5)

        decimals = gu.get_decimals(ax.get_xlim(), ax.get_xticks())
        ax.xaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        gu.standard_tick_styles(ax)

        return ax
