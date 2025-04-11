"""Module for calculating and visualizing transaction statistics by segment.

This module provides the `SegTransactionStats` class, which allows for the computation of
transaction-based statistics grouped by one or more segment columns. The statistics include
aggregations such as total spend, unique customers, transactions per customer, and optional
custom aggregations.

The module supports both Pandas DataFrames and Ibis Tables as input data formats. It also
offers visualization capabilities to generate plots of segment-based statistics.
"""

from typing import Literal

import ibis
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import ColumnHelper
from pyretailscience.style.tailwind import COLORS


class SegTransactionStats:
    """Calculates transaction statistics by segment."""

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        data: pd.DataFrame | ibis.Table,
        segment_col: str | list[str] = "segment_name",
        calc_total: bool = True,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Calculates transaction statistics by segment.

        Args:
            data (pd.DataFrame | ibis.Table): The transaction data. The dataframe must contain the columns
                customer_id, unit_spend and transaction_id. If the dataframe contains the column unit_quantity, then
                the columns unit_spend and unit_quantity are used to calculate the price_per_unit and
                units_per_transaction.
            segment_col (str | list[str], optional): The column or list of columns to use for the segmentation.
                Defaults to "segment_name".
            calc_total (bool, optional): Whether to include the total row. Defaults to True.
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations to perform.
                The keys in the dictionary will be the column names for the aggregation results.
                The values are tuples with (column_name, aggregation_function), where:
                - column_name is the name of the column to aggregate
                - aggregation_function is a string name of an Ibis aggregation function (e.g., "nunique", "sum")
                Example: {"stores": ("store_id", "nunique")} would count unique store_ids.
        """
        cols = ColumnHelper()

        if isinstance(segment_col, str):
            segment_col = [segment_col]
        required_cols = [
            cols.customer_id,
            cols.unit_spend,
            cols.transaction_id,
            *segment_col,
        ]
        if cols.unit_qty in data.columns:
            required_cols.append(cols.unit_qty)

        missing_cols = set(required_cols) - set(data.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        # Validate extra_aggs if provided
        if extra_aggs:
            for col_tuple in extra_aggs.values():
                col, func = col_tuple
                if col not in data.columns:
                    msg = f"Column '{col}' specified in extra_aggs does not exist in the data"
                    raise ValueError(msg)
                if not hasattr(data[col], func):
                    msg = f"Aggregation function '{func}' not available for column '{col}'"
                    raise ValueError(msg)

        self.segment_col = segment_col
        self.extra_aggs = {} if extra_aggs is None else extra_aggs

        self.table = self._calc_seg_stats(data, segment_col, calc_total, self.extra_aggs)

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
    def _calc_seg_stats(
        data: pd.DataFrame | ibis.Table,
        segment_col: list[str],
        calc_total: bool = True,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
    ) -> ibis.Table:
        """Calculates the transaction statistics by segment.

        Args:
            data (pd.DataFrame | ibis.Table): The transaction data.
            segment_col (list[str]): The columns to use for the segmentation.
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations to perform.
            calc_total (bool, optional): Whether to include the total row. Defaults to True.
                The keys in the dictionary will be the column names for the aggregation results.
                The values are tuples with (column_name, aggregation_function).

        Returns:
            pd.DataFrame: The transaction statistics by segment.

        """
        if isinstance(data, pd.DataFrame):
            data = ibis.memtable(data)

        elif not isinstance(data, ibis.Table):
            raise TypeError("data must be either a pandas DataFrame or an ibis Table")

        cols = ColumnHelper()

        # Base aggregations for segments
        aggs = {
            cols.agg_unit_spend: data[cols.unit_spend].sum(),
            cols.agg_transaction_id: data[cols.transaction_id].nunique(),
            cols.agg_customer_id: data[cols.customer_id].nunique(),
        }
        if cols.unit_qty in data.columns:
            aggs[cols.agg_unit_qty] = data[cols.unit_qty].sum()

        # Add extra aggregations if provided
        if extra_aggs:
            for agg_name, col_tuple in extra_aggs.items():
                col, func = col_tuple
                aggs[agg_name] = getattr(data[col], func)()

        # Calculate metrics for segments
        segment_metrics = data.group_by(segment_col).aggregate(**aggs)
        final_metrics = segment_metrics

        if calc_total:
            total_metrics = data.aggregate(**aggs).mutate({col: ibis.literal("Total") for col in segment_col})
            final_metrics = ibis.union(segment_metrics, total_metrics)

        total_customers = data[cols.customer_id].nunique()

        # Cross join with total_customers to make it available for percentage calculation
        final_metrics = final_metrics.mutate(
            **{
                cols.calc_spend_per_cust: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_customer_id],
                cols.calc_spend_per_trans: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_transaction_id],
                cols.calc_trans_per_cust: ibis._[cols.agg_transaction_id] / ibis._[cols.agg_customer_id].cast("float"),
                cols.customers_pct: ibis._[cols.agg_customer_id].cast("float") / total_customers,
            },
        )

        if cols.unit_qty in data.columns:
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc_price_per_unit: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_unit_qty].nullif(0),
                    cols.calc_units_per_trans: ibis._[cols.agg_unit_qty]
                    / ibis._[cols.agg_transaction_id].cast("float"),
                },
            )
        return final_metrics

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the transaction statistics by segment."""
        if self._df is None:
            cols = ColumnHelper()
            col_order = [
                *self.segment_col,
                *SegTransactionStats._get_col_order(include_quantity=cols.agg_unit_qty in self.table.columns),
            ]

            # Add any extra aggregation columns to the column order
            if hasattr(self, "extra_aggs") and self.extra_aggs:
                col_order.extend(self.extra_aggs.keys())

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
            ValueError: If multiple segment columns are used, as plotting is only supported for a single segment column.
        """
        if sort_order not in ["ascending", "descending", None]:
            raise ValueError("sort_order must be either 'ascending' or 'descending' or None")
        if orientation not in ["vertical", "horizontal"]:
            raise ValueError("orientation must be either 'vertical' or 'horizontal'")
        if len(self.segment_col) > 1:
            raise ValueError("Plotting is only supported for a single segment column")

        default_title = f"{value_col.title()} by Segment"
        kind = "bar"
        if orientation == "horizontal":
            kind = "barh"

        # Use the first segment column for plotting
        plot_segment_col = self.segment_col[0]
        val_s = self.df.set_index(plot_segment_col)[value_col]
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
