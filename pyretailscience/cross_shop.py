"""This module contains the CrossShop class that is used to create a cross-shop diagram."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib_set_diagrams import EulerDiagram, VennDiagram

from pyretailscience.data.contracts import CustomContract, build_expected_columns, build_non_null_columns
from pyretailscience.style import graph_utils as gu
from pyretailscience.style.graph_utils import GraphStyles
from pyretailscience.style.tailwind import COLORS


class CrossShop:
    """A class to create a cross-shop diagram."""

    def __init__(
        self,
        df: pd.DataFrame,
        group_1_idx: list[bool] | pd.Series,
        group_2_idx: list[bool] | pd.Series,
        group_3_idx: list[bool] | pd.Series | None = None,
        labels: list[str] | None = None,
        value_col: str = "total_price",
        agg_func: str = "sum",
    ) -> None:
        """Creates a cross-shop diagram that is used to show the overlap of customers between different groups.

        Args:
            df (pd.DataFrame): The dataframe with transactional data.
            group_1_idx (list[bool], pd.Series): A list of bool values determining whether the row is a part of the
                first group.
            group_2_idx (list[bool], pd.Series): A list of bool values determining whether the row is a part of the
                second group.
            group_3_idx (list[bool], pd.Series, optional): An optional list of bool values determining whether the
                row is a part of the third group. Defaults to None. If not supplied, only two groups will be considered.
            labels (list[str], optional): The labels for the groups. Defaults to None.
            value_col (str, optional): The column to aggregate. Defaults to "total_price".
            agg_func (str, optional): The aggregation function. Defaults to "sum".

        Returns:
            None

        Raises:
            ValueError: If the dataframe does not contain the required columns or if the number of labels does not match
                the number of group indexes given.
        """
        required_cols = ["customer_id", value_col]
        contract = CustomContract(
            df,
            basic_expectations=build_expected_columns(columns=required_cols),
            extended_expectations=build_non_null_columns(columns=required_cols),
        )
        if contract.validate() is False:
            msg = f"The dataframe requires the columns {required_cols} and they must be non-null"
            raise ValueError(msg)

        self.group_count = 2 if group_3_idx is None else 3

        if (labels is not None) and (len(labels) != self.group_count):
            raise ValueError("The number of labels must be equal to the number of group indexes given")

        self.labels = labels

        self.cross_shop_df = self._calc_cross_shop(
            df=df,
            group_1_idx=group_1_idx,
            group_2_idx=group_2_idx,
            group_3_idx=group_3_idx,
            value_col=value_col,
            agg_func=agg_func,
        )
        self.cross_shop_table_df = self._calc_cross_shop_table(
            df=self.cross_shop_df,
            value_col=value_col,
        )

    @staticmethod
    def _calc_cross_shop(
        df: pd.DataFrame,
        group_1_idx: list[bool],
        group_2_idx: list[bool],
        group_3_idx: list[bool] | None = None,
        value_col: str = "total_price",
        agg_func: str = "sum",
    ) -> pd.DataFrame:
        """Calculate the cross-shop dataframe that will be used to plot the diagram.

        Args:
            df (pd.DataFrame): The dataframe with transactional data.
            group_1_idx (list[bool]): A list of bool values determining whether the row is a part of the first group.
            group_2_idx (list[bool]): A list of bool values determining whether the row is a part of the second group.
            group_3_idx (list[bool], optional): An optional list of bool values determining whether the row is a part
                of the third group. Defaults to None. If not supplied, only two groups will be considered.
            value_col (str, optional): The column to aggregate. Defaults to "total_price".
            agg_func (str, optional): The aggregation function. Defaults to "sum".

        Returns:
            pd.DataFrame: The cross-shop dataframe.

        Raises:
            ValueError: If the groups are not mutually exclusive.
        """
        if isinstance(group_1_idx, list):
            group_1_idx = pd.Series(group_1_idx)
        if isinstance(group_2_idx, list):
            group_2_idx = pd.Series(group_2_idx)
        if group_3_idx is not None and isinstance(group_3_idx, list):
            group_3_idx = pd.Series(group_3_idx)

        cs_df = df[["customer_id"]].copy()

        cs_df["group_1"] = group_1_idx.astype(int)
        cs_df["group_2"] = group_2_idx.astype(int)
        group_cols = ["group_1", "group_2"]

        if group_3_idx is not None:
            cs_df["group_3"] = group_3_idx.astype(int)
            group_cols += ["group_3"]

        if (cs_df[group_cols].sum(axis=1) > 1).any():
            raise ValueError("The groups must be mutually exclusive.")

        if not any(group_1_idx) or not any(group_2_idx) or (group_3_idx is not None and not any(group_3_idx)):
            raise ValueError("There must at least one row selected for group_1_idx, group_2_idx, and group_3_idx.")

        cs_df = cs_df.groupby("customer_id")[group_cols].max()
        cs_df["groups"] = cs_df[group_cols].apply(lambda x: tuple(x), axis=1)

        kpi_df = df.groupby("customer_id")[value_col].agg(agg_func)

        return cs_df.merge(kpi_df, left_index=True, right_index=True)

    @staticmethod
    def _calc_cross_shop_table(
        df: pd.DataFrame,
        value_col: str = "total_price",
    ) -> pd.DataFrame:
        """Calculate the aggregated cross-shop table that will be used to plot the diagram.

        Args:
            df (pd.DataFrame): The cross-shop dataframe.
            value_col (str, optional): The column to aggregate. Defaults to "total_price".

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
