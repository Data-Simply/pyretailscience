"""This module contains classes for segmenting customers based on their spend and transaction statistics by segment."""

from typing import Literal

import ibis
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import ColumnHelper, get_option
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
            self.df["segment_name"],
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
            df (pd.DataFrame): A dataframe with the customer_id and segment_name columns.

        Raises:
            ValueError: If the dataframe does not have the columns customer_id and segment_name.
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, "segment_name"]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.df = df[[cols.customer_id, "segment_name"]].set_index(cols.customer_id)


class ThresholdSegmentation(BaseSegmentation):
    """Segments customers based on user-defined thresholds and segments."""

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        thresholds: list[float],
        segments: dict[any, str],
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_segment_name: str = "Zero",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
    ) -> None:
        """Segments customers based on user-defined thresholds and segments.

        Args:
            df (pd.DataFrame | ibis.Table): A dataframe with the transaction data. The dataframe must contain a customer_id column.
            thresholds (List[float]): The percentile thresholds for segmentation.
            segments (Dict[str, str]): A dictionary where keys are segment IDs and values are segment names.
            value_col (str, optional): The column to use for the segmentation. Defaults to get_option("column.unit_spend").
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_segment_name (str, optional): The name of the segment for customers with zero spend. Defaults to "Zero".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional): How to handle
                customers with zero spend. Defaults to "separate_segment".

        Raises:
            ValueError: If the dataframe is missing the columns option column.customer_id or `value_col`, or these
                columns contain null values.
        """
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("The thresholds must be unique.")

        if len(thresholds) != len(segments):
            raise ValueError("The number of thresholds must match the number of segments.")

        if isinstance(df, pd.DataFrame):
            df: ibis.Table = ibis.memtable(df)

        value_col = get_option("column.unit_spend") if value_col is None else value_col

        required_cols = [get_option("column.customer_id"), value_col]

        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        df = df.group_by(get_option("column.customer_id")).aggregate(
            **{value_col: getattr(df[value_col], agg_func)()},
        )

        # Separate customers with zero spend
        zero_df = None
        if zero_value_customers == "exclude":
            df = df.filter(df[value_col] != 0)
        elif zero_value_customers == "separate_segment":
            zero_df = df.filter(df[value_col] == 0).mutate(segment_name=ibis.literal(zero_segment_name))
            df = df.filter(df[value_col] != 0)

        window = ibis.window(order_by=ibis.asc(df[value_col]))
        df = df.mutate(ptile=ibis.percent_rank().over(window))

        case = ibis.case()

        for quantile, segment in zip(thresholds, segments, strict=True):
            case = case.when(df["ptile"] <= quantile, segment)

        case = case.end()

        df = df.mutate(segment_name=case).drop(["ptile"])

        if zero_value_customers == "separate_segment":
            df = ibis.union(df, zero_df)

        self.table = df

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names."""
        if self._df is None:
            self._df = self.table.execute().set_index(get_option("column.customer_id"))
        return self._df


class HMLSegmentation(ThresholdSegmentation):
    """Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
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
        thresholds = [0.500, 0.800, 1]
        segments = ["Light", "Medium", "Heavy"]
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

    _df: pd.DataFrame | None = None

    def __init__(self, data: pd.DataFrame | ibis.Table, segment_col: str = "segment_name") -> None:
        """Calculates transaction statistics by segment.

        Args:
            data (pd.DataFrame | ibis.Table): The transaction data. The dataframe must contain the columns
                customer_id, unit_spend and transaction_id. If the dataframe contains the column unit_quantity, then
                the columns unit_spend and unit_quantity are used to calculate the price_per_unit and
                units_per_transaction.
            segment_col (str, optional): The column to use for the segmentation. Defaults to "segment_name".
        """
        cols = ColumnHelper()
        required_cols = [
            cols.customer_id,
            cols.unit_spend,
            cols.transaction_id,
            segment_col,
        ]
        if cols.unit_qty in data.columns:
            required_cols.append(cols.unit_qty)

        missing_cols = set(required_cols) - set(data.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.segment_col = segment_col

        self.table = self._calc_seg_stats(data, segment_col)

    @staticmethod
    def _get_col_order(include_quantity: bool) -> list[str]:
        """Returns the default column order.

        Columns should be supplied in the same order regardless of the function being called.

        Args:
            include_quantity (bool): Whether to include the columns related to quantity.

        Returns:
            list[str]: The default column order.
        """
        cols = ColumnHelper()
        col_order = [
            cols.agg_unit_spend,
            cols.agg_transaction_id,
            cols.agg_customer_id,
            cols.calc_spend_per_cust,
            cols.calc_spend_per_trans,
            cols.calc_trans_per_cust,
            cols.customers_pct,
        ]
        if include_quantity:
            col_order.insert(3, "units")
            col_order.insert(7, cols.calc_units_per_trans)
            col_order.insert(7, cols.calc_price_per_unit)

        return col_order

    @staticmethod
    def _calc_seg_stats(data: pd.DataFrame | ibis.Table, segment_col: str) -> ibis.Table:
        """Calculates the transaction statistics by segment.

        Args:
            data (pd.DataFrame | ibis.Table): The transaction data.
            segment_col (str): The column to use for the segmentation.

        Returns:
            pd.DataFrame: The transaction statistics by segment.

        """
        if isinstance(data, pd.DataFrame):
            data = ibis.memtable(data)

        elif not isinstance(data, ibis.Table):
            raise TypeError("data must be either a pandas DataFrame or a ibis Table")

        cols = ColumnHelper()

        # Base aggregations for segments
        aggs = {
            cols.agg_unit_spend: data[cols.unit_spend].sum(),
            cols.agg_transaction_id: data[cols.transaction_id].nunique(),
            cols.agg_customer_id: data[cols.customer_id].nunique(),
        }
        if cols.unit_qty in data.columns:
            aggs[cols.agg_unit_qty] = data[cols.unit_qty].sum()

        # Calculate metrics for segments and total
        segment_metrics = data.group_by(segment_col).aggregate(**aggs)
        total_metrics = data.group_by(ibis.literal("Total").name(segment_col)).aggregate(**aggs)
        total_customers = data[cols.customer_id].nunique()

        # Cross join with total_customers to make it available for percentage calculation
        final_metrics = ibis.union(segment_metrics, total_metrics).mutate(
            **{
                cols.calc_spend_per_cust: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_customer_id],
                cols.calc_spend_per_trans: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_transaction_id],
                cols.calc_trans_per_cust: ibis._[cols.agg_transaction_id] / ibis._[cols.agg_customer_id],
                cols.customers_pct: ibis._[cols.agg_customer_id] / total_customers,
            },
        )

        if cols.unit_qty in data.columns:
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc_price_per_unit: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_unit_qty].nullif(0),
                    cols.calc_units_per_trans: ibis._[cols.agg_unit_qty] / ibis._[cols.agg_transaction_id],
                },
            )
        return final_metrics

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the transaction statistics by segment."""
        if self._df is None:
            cols = ColumnHelper()
            col_order = [
                self.segment_col,
                *SegTransactionStats._get_col_order(include_quantity=cols.agg_unit_qty in self.table.columns),
            ]
            self._df = self.table.execute()[col_order]
        return self._df

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

        val_s = self.df.set_index(self.segment_col)[value_col]
        if hide_total:
            val_s = val_s[val_s.index != "Total"]

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
