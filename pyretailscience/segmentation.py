from typing import Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.data.contracts import TransactionItemLevelContract, TransactionLevelContract
from pyretailscience.style.tailwind import COLORS


class HMLSegmentation:
    def __init__(self, df: pd.DataFrame, value_col: str = "total_price") -> None:
        """
        Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
                TransactionItemLevelContract or the TransactionLevelContract.
            value_col (str, optional): The column to use for the segmentation. Defaults to "total_price".

        Raises:
            ValueError: If the dataframe does not comply with the TransactionItemLevelContract or
                TransactionLevelContract.
        """

        if TransactionItemLevelContract(df).validate() is False and TransactionLevelContract(df).validate() is False:
            raise ValueError("The dataframe does not comply with the TransactionItemLevelContract")

        # Group by customer_id and calculate total_spend
        grouped_df = df.groupby("customer_id")[value_col].sum().to_frame(value_col)

        # Separate customers with zero spend
        zero_idx = grouped_df[value_col] == 0
        zero_cust_df = grouped_df[zero_idx]
        zero_cust_df["segment_name"] = "Zero"

        hml_df = grouped_df[~zero_idx]

        # Create a new column 'segment' based on the total_spend
        hml_df["segment_name"] = pd.qcut(
            hml_df[value_col],
            q=[0, 0.500, 0.800, 1],
            labels=["Light", "Medium", "Heavy"],
        )

        hml_df = pd.concat([hml_df, zero_cust_df])

        segment_code_map = {"Light": "L", "Medium": "M", "Heavy": "H", "Zero": "Z"}

        hml_df["segment_id"] = hml_df["segment_name"].map(segment_code_map)

        self.df = grouped_df

    def add_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the segment to the dataframe based on the customer_id column.

        Args:
            df (pd.DataFrame): The dataframe to add the segment to. The dataframe must have a customer_id column.

        Returns:
            pd.DataFrame: The dataframe with the segment added.

        Raises:
            ValueError: If the number of rows before and after the merge do not match.
        """

        # TODO: Add a contract that ensures there's a customer ID column or matches one or more of a set of contracts
        # efficently - Eg checks all the quick validations and then tries the extended validations

        rows_before = len(df)
        df = df.merge(self.df[["segment_name", "segment_id"]], how="left", left_on="customer_id", right_index=True)
        rows_after = len(df)
        if rows_before != rows_after:
            raise ValueError("The number of rows before and after the merge do not match. This should not happen.")

        return df


class SegTransactionStats:
    def __init__(self, df: pd.DataFrame, segment_col: str = "segment_id") -> None:
        """
        Calculates transaction statistics by segment.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
                TransactionItemLevelContract or the TransactionLevelContract.
            segment_col (str, optional): The column to use for the segmentation. Defaults to "segment_id".

        Raises:
            NotImplementedError: If the dataframe does not comply with the TransactionItemLevelContract or
                TransactionLevelContract.

        """
        self.segment_col = segment_col
        if TransactionItemLevelContract(df).validate() is True:
            stats_df = df.groupby(segment_col).agg(
                revenue=("total_price", "sum"),
                transactions=("transaction_id", "nunique"),
                customers=("customer_id", "nunique"),
                total_quantity=("quantity", "sum"),
            )
            stats_df["price_per_unit"] = stats_df["revenue"] / stats_df["total_quantity"]
            stats_df["quantity_per_transaction"] = stats_df["total_quantity"] / stats_df["transactions"]
        elif TransactionLevelContract(df).validate() is True:
            stats_df = df.groupby(segment_col).agg(
                revenue=("total_price", "sum"),
                transactions=("transaction_id", "nunique"),
                customers=("customer_id", "nunique"),
            )
        else:
            raise NotImplementedError(
                "The dataframe does not comply with the TransactionItemLevelContract or TransactionLevelContract. "
                "These are the only two contracts supported at this time."
            )
        total_num_customers = df["customer_id"].nunique()
        stats_df["spend_per_cust"] = stats_df["revenue"] / stats_df["customers"]
        stats_df["spend_per_transaction"] = stats_df["revenue"] / stats_df["transactions"]
        stats_df["transactions_per_customer"] = stats_df["transactions"] / stats_df["customers"]
        stats_df["customers_pct"] = stats_df["customers"] / total_num_customers

        self.df = stats_df

    def plot(
        self,
        value_col: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        sort_order: Literal["ascending", "descending", None] = None,
        source_text: str = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """
        Plots the value_col by segment.

        Args:
            value_col (str): The column to plot.
            title (str, optional): The title of the plot. Defaults to None.
            x_label (str, optional): The x-axis label. Defaults to None. When None the x-axis label is blank when the
                orientation is horizontal. When the orientation is vertical it is set to the `value_col` in title case.
            y_label (str, optional): The y-axis label. Defaults to None. When None the y-axis label is set to the
                `value_col` in title case when the orientation is horizontal. Then the orientation is vertical it is
                set to blank
            ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
            orientation (Literal["vertical", "horizontal"], optional): The orientation of the plot. Defaults to
                "vertical".
            sort_order (Literal["ascending", "descending", None], optional): The sort order of the segments.
                Defaults to None. If None, the segments are plotted in the order they appear in the dataframe.
            source_text (str, optional): The source text to add to the plot. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Pandas plot function.

        Returns:
            SubplotBase: The matplotlib axes object.

        Raises:
            ValueError: If the sort_order is not "ascending", "descending" or None.
            ValueError: If the orientation is not "vertical" or "horizontal".
        """

        if sort_order not in ["ascending", "descending", None]:
            raise ValueError("sort_order must be either 'ascending' or 'descending' or None")
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("orientation must be either 'vertical' or 'horizontal'")

        default_title = f"{value_col.title()} by Segment"
        kind = "bar"
        if orientation == "horizontal":
            kind = "barh"

        val_s = self.df[value_col]
        if sort_order is not None:
            ascending = sort_order == "ascending"
            val_s = val_s.sort_values(ascending=ascending)

        ax = val_s.plot(
            kind=kind,
            color=COLORS["green"][500],
            legend=False,
            ax=ax,
            **kwargs,
        )
        ax = gu.standard_graph_styles(ax)

        if orientation == "vertical":
            plot_y_label = gu.not_none(y_label, value_col.title())
            plot_x_label = gu.not_none(x_label, "")
            decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())
            ax.yaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))
        else:
            plot_y_label = gu.not_none(y_label, "")
            plot_x_label = gu.not_none(x_label, value_col.title())
            decimals = gu.get_decimals(ax.get_xlim(), ax.get_xticks())
            ax.xaxis.set_major_formatter(lambda x, pos: gu.human_format(x, pos, decimals=decimals))

        ax.set_title(gu.not_none(title, default_title))
        ax.set_ylabel(plot_y_label)
        ax.set_xlabel(plot_x_label)

        if source_text is not None:
            ax.annotate(
                source_text,
                xy=(-0.1, -0.2),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=10,
            )

        return ax
