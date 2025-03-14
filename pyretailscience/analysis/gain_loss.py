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
"""

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.style.tailwind import COLORS


class GainLoss:
    """A class to perform gain loss analysis on a DataFrame to assess customer movement between brands or products over time."""

    def __init__(
        self,
        df: pd.DataFrame,
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
            df (pd.DataFrame): The DataFrame to calculate the gain loss table from.
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
        # # Ensure no overlap between p1 and p2
        if not df[p1_index].index.intersection(df[p2_index].index).empty:
            raise ValueError("p1_index and p2_index should not overlap")

        if not df[focus_group_index].index.intersection(df[comparison_group_index].index).empty:
            raise ValueError("focus_group_index and comparison_group_index should not overlap")

        if not len(p1_index) == len(p2_index) == len(focus_group_index) == len(comparison_group_index):
            raise ValueError(
                "p1_index, p2_index, focus_group_index, and comparison_group_index should have the same length",
            )

        required_cols = [get_option("column.customer_id"), value_col] + ([group_col] if group_col is not None else [])
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.focus_group_name = focus_group_name
        self.comparison_group_name = comparison_group_name
        self.group_col = group_col
        self.value_col = value_col

        self.gain_loss_df = self._calc_gain_loss(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            comparison_group_index=comparison_group_index,
            group_col=group_col,
            value_col=value_col,
            agg_func=agg_func,
        )
        self.gain_loss_table_df = self._calc_gains_loss_table(
            gain_loss_df=self.gain_loss_df,
            group_col=group_col,
        )

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
        df: pd.DataFrame,
        p1_index: list[bool],
        p2_index: list[bool],
        focus_group_index: list[bool],
        comparison_group_index: list,
        group_col: str | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """Calculate the gain loss table for a given DataFrame at the customer level.

        Args:
            df (pd.DataFrame): The DataFrame to calculate the gain loss table from.
            p1_index (list[bool]): The index for the first time period.
            p2_index (list[bool]): The index for the second time period.
            focus_group_index (list[bool]): The index for the focus group.
            comparison_group_index (list[bool]): The index for the comparison group.
            group_col (str | None, optional): The column to group by. Defaults to None.
            value_col (str, optional): The column to calculate the gain loss from. Defaults to option column.unit_spend.
            agg_func (str, optional): The aggregation function to use. Defaults to "sum".

        Returns:
            pd.DataFrame: The gain loss table.
        """
        cols = ColumnHelper()
        df = df[p1_index | p2_index].copy()
        df[cols.customer_id] = df[cols.customer_id].astype("category")

        grp_cols = [cols.customer_id] if group_col is None else [group_col, cols.customer_id]

        p1_df = pd.concat(
            [
                df[focus_group_index & p1_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[comparison_group_index & p1_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[(focus_group_index | comparison_group_index) & p1_index]
                .groupby(grp_cols, observed=False)[value_col]
                .agg(agg_func),
            ],
            axis=1,
        )
        p1_df.columns = ["focus", "comparison", "total"]

        p2_df = pd.concat(
            [
                df[focus_group_index & p2_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[comparison_group_index & p2_index].groupby(grp_cols, observed=False)[value_col].agg(agg_func),
                df[(focus_group_index | comparison_group_index) & p2_index]
                .groupby(grp_cols, observed=False)[value_col]
                .agg(agg_func),
            ],
            axis=1,
        )
        p2_df.columns = ["focus", "comparison", "total"]

        gl_df = p1_df.merge(p2_df, on=grp_cols, how="outer", suffixes=("_p1", "_p2")).fillna(0)

        # Remove rows that are all 0 due to grouping by customer_id as a categorical with observed=False
        gl_df = gl_df[~(gl_df == 0).all(axis=1)]

        gl_df["focus_diff"] = gl_df["focus_p2"] - gl_df["focus_p1"]
        gl_df["comparison_diff"] = gl_df["comparison_p2"] - gl_df["comparison_p1"]
        gl_df["total_diff"] = gl_df["total_p2"] - gl_df["total_p1"]

        (
            gl_df["new"],
            gl_df["lost"],
            gl_df["increased_focus"],
            gl_df["decreased_focus"],
            gl_df["switch_from_comparison"],
            gl_df["switch_to_comparison"],
        ) = zip(
            *gl_df.apply(
                lambda x: GainLoss.process_customer_group(
                    focus_p1=x["focus_p1"],
                    comparison_p1=x["comparison_p1"],
                    focus_p2=x["focus_p2"],
                    comparison_p2=x["comparison_p2"],
                    focus_diff=x["focus_diff"],
                    comparison_diff=x["comparison_diff"],
                ),
                axis=1,
            ),
            strict=False,
        )

        return gl_df

    @staticmethod
    def _calc_gains_loss_table(
        gain_loss_df: pd.DataFrame,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """Aggregates the gain loss table to show the total gains and losses across customers.

        Args:
            gain_loss_df (pd.DataFrame): The gain loss table at customer level to aggregate.
            group_col (str | None, optional): The column to group by. Defaults to None.

        Returns:
            pd.DataFrame: The aggregated gain loss table
        """
        if group_col is None:
            return gain_loss_df.sum().to_frame("").T

        return gain_loss_df.groupby(level=0).sum()

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
        """Plot the gain loss table.

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

        if self.group_col is None:
            ax = self.gain_loss_table_df[increase_cols].plot.barh(stacked=True, color=green_colors, ax=ax, **kwargs)
            self.gain_loss_table_df[decrease_cols].plot.barh(stacked=True, ax=ax, color=red_colors)
            default_y_label = self.focus_group_name

        else:
            ax = self.gain_loss_table_df[increase_cols].plot.barh(stacked=True, color=green_colors, **kwargs)
            self.gain_loss_table_df[decrease_cols].plot.barh(stacked=True, ax=ax, color=red_colors)
            default_y_label = self.group_col

        legend_bbox_to_anchor = None
        if move_legend_outside:
            legend_bbox_to_anchor = (1.05, 1)

        # TODO: Ensure that each label actually has data before adding to the legend
        legend = ax.legend(
            [
                "New",
                f"Increased {self.focus_group_name}",
                f"Switch From {self.comparison_group_name}",
                "Lost",
                f"Decreased {self.focus_group_name}",
                f"Switch To {self.comparison_group_name}",
            ],
            frameon=True,
            bbox_to_anchor=legend_bbox_to_anchor,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("white")

        ax = gu.standard_graph_styles(
            ax,
            title=gu.not_none(title, f"Gain Loss from {self.focus_group_name} to {self.comparison_group_name}"),
            y_label=gu.not_none(y_label, default_y_label),
            x_label=gu.not_none(x_label, self.value_col),
            move_legend_outside=move_legend_outside,
        )

        decimals = gu.get_decimals(ax.get_xlim(), ax.get_xticks())
        ax.xaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))

        ax.axvline(0, color="black", linewidth=0.5)

        if source_text is not None:
            gu.add_source_text(ax=ax, source_text=source_text)

        gu.standard_tick_styles(ax)

        return ax
