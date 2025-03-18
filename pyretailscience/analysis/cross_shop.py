"""This module contains the CrossShop class that is used to create a cross-shop diagram."""

import ibis
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib_set_diagrams import EulerDiagram, VennDiagram

from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.style import graph_utils as gu
from pyretailscience.style.graph_utils import GraphStyles
from pyretailscience.style.tailwind import COLORS


class CrossShop:
    """A class to create a cross-shop diagram."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        group_1_col: str,
        group_1_val: str,
        group_2_col: str,
        group_2_val: str,
        group_3_col: str | None = None,
        group_3_val: str | None = None,
        labels: list[str] | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> None:
        """Creates a cross-shop diagram that is used to show the overlap of customers between different groups.

        Args:
            df (pd.DataFrame | ibis.Table):  The input DataFrame or ibis Table containing transactional data.
            group_1_col (str): The column name for the first group.
            group_1_val (str): The value of the first group to match.
            group_2_col (str): The column name for the second group.
            group_2_val (str): The value of the second group to match.
            group_3_col (str, optional): The column name for the third group. Defaults to None.
            group_3_val (str, optional): The value of the third group to match. Defaults to None.
            labels (list[str], optional): The labels for the groups. Defaults to None.
            value_col (str, optional): The column to aggregate. Defaults to the option column.unit_spend.
            agg_func (str, optional): The aggregation function. Defaults to "sum".

        Returns:
            None

        Raises:
            ValueError: If the dataframe does not contain the required columns or if the number of labels does not match
                the number of group indexes given.
        """
        required_cols = [get_option("column.customer_id"), value_col]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.group_count = 2 if group_3_col is None else 3

        if (labels is not None) and (len(labels) != self.group_count):
            raise ValueError("The number of labels must be equal to the number of group indexes given")

        self.labels = labels

        self.cross_shop_df = self._calc_cross_shop(
            df=df,
            group_1_col=group_1_col,
            group_1_val=group_1_val,
            group_2_col=group_2_col,
            group_2_val=group_2_val,
            group_3_col=group_3_col,
            group_3_val=group_3_val,
            value_col=value_col,
            agg_func=agg_func,
        )
        self.cross_shop_table_df = self._calc_cross_shop_table(
            df=self.cross_shop_df,
            value_col=value_col,
        )

    @staticmethod
    def _calc_cross_shop(
        df: pd.DataFrame | ibis.Table,
        group_1_col: str,
        group_1_val: str,
        group_2_col: str,
        group_2_val: str,
        group_3_col: str | None = None,
        group_3_val: str | None = None,
        value_col: str = get_option("column.unit_spend"),
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """Calculate the cross-shop dataframe that will be used to plot the diagram.

        Args:
            df (pd.DataFrame | ibis.Table):  The input DataFrame or ibis Table containing transactional data.
            group_1_col (str): Column name for the first group.
            group_1_val (str): Value to filter for the first group.
            group_2_col (str): Column name for the second group.
            group_2_val (str): Value to filter for the second group.
            group_3_col (str, optional): Column name for the third group. Defaults to None.
            group_3_val (str, optional): Value to filter for the third group. Defaults to None.
            value_col (str, optional): The column to aggregate. Defaults to option column.unit_spend.
            agg_func (str, optional): The aggregation function. Defaults to "sum".

        Returns:
            pd.DataFrame: The cross-shop dataframe.

        Raises:
            ValueError: If group_3_col or group_3_val is populated, then the other must be as well.
        """
        cols = ColumnHelper()

        if isinstance(df, pd.DataFrame):
            df: ibis.Table = ibis.memtable(df)
        if (group_3_col is None) != (group_3_val is None):
            raise ValueError("If group_3_col or group_3_val is populated, then the other must be as well")

        # Using a temporary value column to avoid duplicate column errors during selection. This happens when `value_col` has the same name as `customer_id`, causing conflicts in `.select()`.
        temp_value_col = "temp_value_col"
        df = df.mutate(**{temp_value_col: df[value_col]})

        group_1 = (df[group_1_col] == group_1_val).cast("int32").name("group_1")
        group_2 = (df[group_2_col] == group_2_val).cast("int32").name("group_2")
        group_3 = (df[group_3_col] == group_3_val).cast("int32").name("group_3") if group_3_col else None

        group_cols = ["group_1", "group_2"]
        select_cols = [df[cols.customer_id], group_1, group_2]
        if group_3 is not None:
            group_cols.append("group_3")
            select_cols.append(group_3)

        cs_df = df.select([*select_cols, df[temp_value_col]]).order_by(cols.customer_id)
        cs_df = (
            cs_df.group_by(cols.customer_id)
            .aggregate(
                **{col: cs_df[col].max().name(col) for col in group_cols},
                **{temp_value_col: getattr(cs_df[temp_value_col], agg_func)().name(temp_value_col)},
            )
            .order_by(cols.customer_id)
        ).execute()

        cs_df["groups"] = cs_df[group_cols].apply(lambda x: tuple(x), axis=1)
        column_order = [cols.customer_id, *group_cols, "groups", temp_value_col]
        cs_df = cs_df[column_order]
        cs_df.set_index(cols.customer_id, inplace=True)
        return cs_df.rename(columns={temp_value_col: value_col})

    @staticmethod
    def _calc_cross_shop_table(
        df: pd.DataFrame,
        value_col: str = get_option("column.unit_spend"),
    ) -> pd.DataFrame:
        """Calculate the aggregated cross-shop table that will be used to plot the diagram.

        Args:
            df (pd.DataFrame): The cross-shop dataframe.
            value_col (str, optional): The column to aggregate. Defaults to option column.unit_spend.

        Returns:
            pd.DataFrame: The cross-shop table.
        """
        df = df.groupby(["groups"], dropna=False)[value_col].sum().reset_index().copy()
        df["percent"] = df[value_col] / df[value_col].sum()
        return df

    @staticmethod
    def translate_text_outward(
        text: plt.Text,
        center_x: float = 0.5,
        center_y: float = 0.5,
        displacement: float = 0.1,
    ) -> None:
        """A helper method that translates the text away from the center of the plot.

        Args:
            text (plt.Text): The text to translate.
            center_x (float, optional): The x-coordinate of the center of the plot. Defaults to 0.5.
            center_y (float, optional): The y-coordinate of the center of the plot. Defaults to 0.5.
            displacement (float, optional): The amount to translate the text by. Defaults to 0.1.

        Returns:
            None
        """
        x, y = text.get_position()
        direction_x, direction_y = x - center_x, y - center_y
        scale = displacement / (direction_x**2 + direction_y**2) ** 0.5
        new_x, new_y = x + scale * direction_x, y + scale * direction_y
        text.set_position((new_x, new_y))

    def plot(
        self,
        title: str | None = None,
        source_text: str | None = None,
        vary_size: bool = False,
        figsize: tuple[int, int] | None = None,
        ax: Axes | None = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the cross-shop diagram.

        Args:
            title (str, optional): The title of the plot. Defaults to None.
            source_text (str, optional): The source text for the plot. Defaults to None.
            vary_size (bool, optional): Whether to vary the size of the circles based on their values. Defaults to
                False.
            figsize (tuple[int, int], optional): The size of the plot. Defaults to None.
            ax (Axes, optional): The axes to plot on. Defaults to None.
            **kwargs (dict[str, any]): Additional keyword arguments to pass to the diagram.

        Returns:
            SubplotBase: The axes of the plot.
        """
        three_circles = 3

        zero_group = (0, 0)
        colors = [COLORS["green"][500], COLORS["green"][800]]
        if self.group_count == three_circles:
            zero_group = (0, 0, 0)
            colors += [COLORS["green"][200]]

        zero_row_idx = self.cross_shop_table_df["groups"] == zero_group
        percent_s = self.cross_shop_table_df[~zero_row_idx].set_index("groups")["percent"]

        subset_labels = percent_s.apply(lambda x: f"{x:.1%}").to_dict()
        subset_sizes = percent_s.to_dict()

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if vary_size:
            diagram = EulerDiagram(
                set_labels=self.labels,
                subset_sizes=subset_sizes,
                subset_labels=subset_labels,
                set_colors=colors,
                ax=ax,
                **kwargs,
            )
        else:
            diagram = VennDiagram(
                set_labels=self.labels,
                subset_sizes=subset_sizes,
                subset_labels=subset_labels,
                set_colors=colors,
                ax=ax,
                **kwargs,
            )

        for text in diagram.set_label_artists:
            text.set_fontproperties(GraphStyles.POPPINS_REG)
            text.set_fontsize(GraphStyles.DEFAULT_AXIS_LABEL_FONT_SIZE)
            if self.group_count == three_circles and not vary_size:
                # Increase the spacing between text and the diagram to avoid overlap
                self.translate_text_outward(text)

        for subset_id in subset_sizes:
            if subset_id not in diagram.subset_label_artists:
                continue
            text = diagram.subset_label_artists[subset_id]
            text.set_fontproperties(GraphStyles.POPPINS_REG)

        if title is not None:
            ax.set_title(
                title,
                fontproperties=GraphStyles.POPPINS_SEMI_BOLD,
                fontsize=GraphStyles.DEFAULT_TITLE_FONT_SIZE,
                pad=GraphStyles.DEFAULT_TITLE_PAD + 20,
            )

        if source_text is not None:
            # Hide the xticks to remove space between the diagram and source text
            ax.set_xticklabels([], visible=False)
            gu.add_source_text(ax=ax, source_text=source_text)

        return ax
