"""This module contains classes for segmenting customers based on their spend and transaction statistics by segment."""

from typing import Literal

import duckdb
import pandas as pd
from duckdb import DuckDBPyRelation
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.data.contracts import (
    CustomContract,
    build_expected_columns,
    build_expected_unique_columns,
    build_non_null_columns,
)
from pyretailscience.options import get_option
from pyretailscience.style.tailwind import COLORS


class BaseSegmentation:
    """A base class for customer segmentation."""

    def add_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the segment to the dataframe based on the customer_id column.

        Args:
            df (pd.DataFrame): The dataframe to add the segment to. The dataframe must have a customer_id column.

        Returns:
            pd.DataFrame: The dataframe with the segment added.

        Raises:
            ValueError: If the number of rows before and after the merge do not match.
        """
        rows_before = len(df)
        df = df.merge(
            self.df[["segment_name", "segment_id"]],
            how="left",
            left_on=get_option("column.customer_id"),
            right_index=True,
        )
        rows_after = len(df)
        if rows_before != rows_after:
            raise ValueError("The number of rows before and after the merge do not match. This should not happen.")

        return df


class ExistingSegmentation(BaseSegmentation):
    """Segments customers based on an existing segment in the dataframe."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Segments customers based on an existing segment in the dataframe.

        Args:
            df (pd.DataFrame): A dataframe with the customer_id, segment_name and segment_id columns.

        Raises:
            ValueError: If the dataframe does not have the columns customer_id, segment_name and segment_id.
        """
        required_cols = get_option("column.customer_id"), "segment_name", "segment_id"
        contract = CustomContract(
            df,
            basic_expectations=build_expected_columns(columns=required_cols),
            extended_expectations=build_non_null_columns(columns=required_cols)
            + build_expected_unique_columns(columns=[required_cols]),
        )

        if contract.validate() is False:
            msg = f"The dataframe requires the columns {required_cols} and they must be non-null and unique."
            raise ValueError(msg)

        self.df = df[[get_option("column.customer_id"), "segment_name", "segment_id"]].set_index(
            get_option("column.customer_id"),
        )


class ThresholdSegmentation(BaseSegmentation):
    """Segments customers based on user-defined thresholds and segments."""

    def __init__(
        self,
        df: pd.DataFrame,
        thresholds: list[float],
        segments: dict[any, str],
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_segment_name: str = "Zero",
        zero_segment_id: str = "Z",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
    ) -> None:
        """Segments customers based on user-defined thresholds and segments.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must contain a customer_id column.
            thresholds (List[float]): The percentile thresholds for segmentation.
            segments (Dict[str, str]): A dictionary where keys are segment IDs and values are segment names.
            value_col (str, optional): The column to use for the segmentation. Defaults to get_option("column.unit_spend").
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_segment_name (str, optional): The name of the segment for customers with zero spend. Defaults to "Zero".
            zero_segment_id (str, optional): The ID of the segment for customers with zero spend. Defaults to "Z".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional): How to handle
                customers with zero spend. Defaults to "separate_segment".

        Raises:
            ValueError: If the dataframe is missing the columns "customer_id" or `value_col`, or these columns contain
                null values.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        value_col = get_option("column.unit_spend") if value_col is None else value_col

        required_cols = [get_option("column.customer_id"), value_col]
        contract = CustomContract(
            df,
            basic_expectations=build_expected_columns(columns=required_cols),
            extended_expectations=build_non_null_columns(columns=required_cols),
        )

        if contract.validate() is False:
            msg = f"The dataframe requires the columns {required_cols} and they must be non-null"
            raise ValueError(msg)

        if len(df) < len(thresholds):
            msg = f"There are {len(df)} customers, which is less than the number of segment thresholds."
            raise ValueError(msg)

        if set(thresholds) != set(thresholds):
            raise ValueError("The thresholds must be unique.")

        thresholds = sorted(thresholds)
        if thresholds[0] != 0:
            thresholds = [0, *thresholds]
        if thresholds[-1] != 1:
            thresholds.append(1)

        if len(thresholds) - 1 != len(segments):
            raise ValueError("The number of thresholds must match the number of segments.")

        # Group by customer_id and calculate total_spend
        grouped_df = df.groupby(get_option("column.customer_id"))[value_col].agg(agg_func).to_frame(value_col)

        # Separate customers with zero spend
        self.df = grouped_df
        if zero_value_customers in ["separate_segment", "exclude"]:
            zero_idx = grouped_df[value_col] == 0
            zero_cust_df = grouped_df[zero_idx].copy()
            zero_cust_df["segment_name"] = zero_segment_name
            zero_cust_df["segment_id"] = zero_segment_id

            self.df = grouped_df[~zero_idx].copy()

        # Create a new column 'segment' based on the total_spend
        labels = list(segments.values())

        self.df["segment_name"] = pd.qcut(
            self.df[value_col],
            q=thresholds,
            labels=labels,
        )

        self.df["segment_id"] = self.df["segment_name"].map({v: k for k, v in segments.items()})

        if zero_value_customers == "separate_segment":
            self.df = pd.concat([self.df, zero_cust_df])


class HMLSegmentation(ThresholdSegmentation):
    """Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend."""

    def __init__(
        self,
        df: pd.DataFrame,
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
    ) -> None:
        """Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend.

        HMLSegmentation is a subclass of ThresholdSegmentation and based around an industry standard definition. The
        thresholds for Heavy (top 20%), Medium (next 30%) and Light (bottom 50%) are chosen based on the pareto
        distribution, commonly know as the 80/20 rule. It is typically used in retail to segment customers based on
        their spend, transaction volume or quantities purchased.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must contain a customer_id column.
            value_col (str, optional): The column to use for the segmentation. Defaults to get_option("column.unit_spend").
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional): How to handle
                customers with zero spend. Defaults to "separate_segment".
        """
        thresholds = [0, 0.500, 0.800, 1]
        segments = {"L": "Light", "M": "Medium", "H": "Heavy"}
        super().__init__(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
        )


class SegTransactionStats:
    """Calculates transaction statistics by segment."""

    def __init__(self, data: pd.DataFrame | DuckDBPyRelation, segment_col: str = "segment_id") -> None:
        """Calculates transaction statistics by segment.

        Args:
            data (pd.DataFrame | DuckDBPyRelation): The transaction data. The dataframe must contain the columns
                customer_id, unit_spend and transaction_id. If the dataframe contains the column unit_quantity, then
                the columns unit_spend and unit_quantity are used to calculate the price_per_unit and
                units_per_transaction.
            segment_col (str, optional): The column to use for the segmentation. Defaults to "segment_id".

        Raises:
            NotImplementedError: If the dataframe does not comply with the TransactionItemLevelContract or
                TransactionLevelContract.

        """
        required_cols = [
            get_option("column.customer_id"),
            get_option("column.unit_spend"),
            get_option("column.transaction_id"),
            segment_col,
        ]
        if get_option("column.unit_quantity") in data.columns:
            required_cols.append(get_option("column.unit_quantity"))

        missing_cols = [col for col in required_cols if col not in data.columns]

        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.segment_col = segment_col

        self.df = self._calc_seg_stats(data, segment_col)

    @staticmethod
    def _calc_seg_stats(data: pd.DataFrame | DuckDBPyRelation, segment_col: str) -> pd.DataFrame:
        """Calculates the transaction statistics by segment.

        Args:
            data (DuckDBPyRelation): The transaction data.
            segment_col (str): The column to use for the segmentation.

        Returns:
            pd.DataFrame: The transaction statistics by segment.

        """
        if isinstance(data, pd.DataFrame):
            data = duckdb.from_df(data)

        base_aggs = [
            f"SUM({get_option('column.unit_spend')}) as {get_option('column.agg.unit_spend')},",
            f"COUNT(DISTINCT {get_option('column.transaction_id')}) as {get_option('column.agg.transaction_id')},",
            f"COUNT(DISTINCT {get_option('column.customer_id')}) as {get_option('column.agg.customer_id')},",
        ]

        total_customers = data.aggregate("COUNT(DISTINCT customer_id)").fetchone()[0]
        return_cols = [
            "*,",
            f"{get_option('column.agg.unit_spend')} / {get_option('column.agg.customer_id')} ",
            f"as {get_option('column.calc.spend_per_customer')},",
            f"{get_option('column.agg.unit_spend')} / {get_option('column.agg.transaction_id')} ",
            f"as {get_option('column.calc.spend_per_transaction')},",
            f"{get_option('column.agg.transaction_id')} / {get_option('column.agg.customer_id')} ",
            f"as {get_option('column.calc.transactions_per_customer')},",
            f"{get_option('column.agg.customer_id')} / {total_customers}",
            f"as customers_{get_option('column.suffix.percent')},",
        ]

        if get_option("column.unit_quantity") in data.columns:
            base_aggs.append(
                f"SUM({get_option('column.unit_quantity')})::bigint as {get_option('column.agg.unit_quantity')},",
            )
            return_cols.extend(
                [
                    f"({get_option('column.agg.unit_spend')} / {get_option('column.agg.unit_quantity')}) ",
                    f"as {get_option('column.calc.price_per_unit')},",
                    f"({get_option('column.agg.unit_quantity')} / {get_option('column.agg.transaction_id')}) ",
                    f"as {get_option('column.calc.units_per_transaction')},",
                ],
            )

        segment_stats = data.aggregate(f"{segment_col} as segment_name," + "".join(base_aggs))
        total_stats = data.aggregate("'Total' as segment_name," + "".join(base_aggs))
        final_stats_df = segment_stats.union(total_stats).select("".join(return_cols)).df()
        final_stats_df = final_stats_df.set_index("segment_name").sort_index()

        # Make sure Total is the last row
        desired_index_sort = final_stats_df.index.drop("Total").tolist() + ["Total"]  # noqa: RUF005

        return final_stats_df.reindex(desired_index_sort)

    def plot(
        self,
        value_col: str,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        sort_order: Literal["ascending", "descending", None] = None,
        source_text: str | None = None,
        hide_total: bool = True,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plots the value_col by segment.

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
            hide_total (bool, optional): Whether to hide the total row. Defaults to True.
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
        if hide_total:
            val_s = val_s[val_s.index != "total"]

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

        ax = gu.standard_graph_styles(
            ax,
            title=gu.not_none(title, default_title),
            x_label=plot_x_label,
            y_label=plot_y_label,
        )

        if source_text is not None:
            gu.add_source_text(ax=ax, source_text=source_text)

        gu.standard_tick_styles(ax)

        return ax
