import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.data.contracts import CustomContract, build_expected_columns, build_non_null_columns
from pyretailscience.style.graph_utils import GraphStyles as gs
from pyretailscience.style.tailwind import COLORS

# TODO: Consider simplifying this by reducing the color range in the get_linear_cmap function.
COLORMAP_MIN = 0.25
COLORMAP_MAX = 0.75


class GainLoss:
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
        value_col: str = "total_price",
        agg_func: str = "sum",
    ):
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
            value_col (str, optional): The column to calculate the gain loss from. Defaults to "total_price".
            agg_func (str, optional): The aggregation function to use. Defaults to "sum".
        """
        # # Ensure no overlap between p1 and p2
        if not df[p1_index].index.intersection(df[p2_index].index).empty:
            raise ValueError("p1_index and p2_index should not overlap")

        if not df[focus_group_index].index.intersection(df[comparison_group_index].index).empty:
            raise ValueError("focus_group_index and comparison_group_index should not overlap")

        if not len(p1_index) == len(p2_index) == len(focus_group_index) == len(comparison_group_index):
            raise ValueError(
                "p1_index, p2_index, focus_group_index, and comparison_group_index should have the same length"
            )

        required_cols = ["customer_id", value_col] + ([group_col] if group_col is not None else [])
        contract = CustomContract(
            df,
            basic_expectations=build_expected_columns(columns=required_cols),
            extended_expectations=build_non_null_columns(columns=required_cols),
        )
        if contract.validate() is False:
            raise ValueError(f"The dataframe requires the columns {required_cols} and they must be non-null")

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
    def _calc_gain_loss(
        df: pd.DataFrame,
        p1_index: list[bool],
        p2_index: list[bool],
        focus_group_index: list[bool],
        comparison_group_index: list,
        group_col: str | None = None,
        value_col: str = "total_price",
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """
        Calculate the gain loss table for a given DataFrame at the customer level.

        Args:
            df (pd.DataFrame): The DataFrame to calculate the gain loss table from.
            p1_index (list[bool]): The index for the first time period.
            p2_index (list[bool]): The index for the second time period.
            focus_group_index (list[bool]): The index for the focus group.
            comparison_group_index (list[bool]): The index for the comparison group.
            group_col (str | None, optional): The column to group by. Defaults to None.
            value_col (str, optional): The column to calculate the gain loss from. Defaults to "total_price".

        Returns:
            pd.DataFrame: The gain loss table.
        """
        df = df[p1_index | p2_index].copy()
        df["customer_id"] = df["customer_id"].astype("category")

        grp_cols = ["customer_id"] if group_col is None else [group_col, "customer_id"]

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

        gl_df["switch_from_comparison"] = (gl_df["focus_diff"] - gl_df["total_diff"]).apply(lambda x: max(x, 0))
        gl_df["switch_to_comparison"] = (gl_df["focus_diff"] - gl_df["total_diff"]).apply(lambda x: min(x, 0))

        gl_df["new"] = gl_df.apply(lambda x: max(x["total_diff"], 0) if x["focus_p1"] == 0 else 0, axis=1)
        gl_df["lost"] = gl_df.apply(lambda x: min(x["total_diff"], 0) if x["focus_p2"] == 0 else 0, axis=1)

        gl_df["increased_focus"] = gl_df.apply(lambda x: max(x["total_diff"], 0) if x["focus_p1"] != 0 else 0, axis=1)
        gl_df["decreased_focus"] = gl_df.apply(lambda x: min(x["total_diff"], 0) if x["focus_p2"] != 0 else 0, axis=1)

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
        else:
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

        # TODO: Ensure that each label ctually has data before adding to the legend
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
        )

        decimals = gu.get_decimals(ax.get_xlim(), ax.get_xticks())
        ax.xaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))

        ax.axvline(0, color="black", linewidth=0.5)

        if source_text is not None:
            ax.annotate(
                source_text,
                xy=(-0.1, -0.2),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=gs.DEFAULT_SOURCE_FONT_SIZE,
                fontproperties=gs.POPPINS_LIGHT_ITALIC,
                color="dimgray",
            )

        # Set the font properties for the tick labels
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(gs.POPPINS_REG)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(gs.POPPINS_REG)

        return ax
