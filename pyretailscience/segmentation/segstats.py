"""Module for calculating and visualizing transaction statistics by segment.

This module provides the `SegTransactionStats` class, which allows for the computation of
transaction-based statistics grouped by one or more segment columns. The statistics include
aggregations such as total spend, unique customers, transactions per customer, and optional
custom aggregations.

The module supports both Pandas DataFrames and Ibis Tables as input data formats. It also
offers visualization capabilities to generate plots of segment-based statistics.
"""

from typing import Any, Literal

import ibis
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.options import ColumnHelper
from pyretailscience.plots.styles.tailwind import COLORS


class SegTransactionStats:
    """Calculates transaction statistics by segment."""

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        data: pd.DataFrame | ibis.Table,
        segment_col: str | list[str] = "segment_name",
        calc_total: bool = True,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
        calc_rollup: bool = False,
        rollup_value: Any | list[Any] = "Total",  # noqa: ANN401 - Any is required for ibis.literal typing
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
            calc_rollup (bool, optional): Whether to calculate rollup totals. Defaults to False.
            rollup_value (Any | list[Any], optional): The value to use for rollup totals. Can be a single value
                applied to all columns or a list of values matching the length of segment_col, with each value
                cast to match the corresponding column type. Defaults to "Total".
        """
        # Convert data to ibis.Table if it's a pandas DataFrame
        if isinstance(data, pd.DataFrame):
            data = ibis.memtable(data)
        elif not isinstance(data, ibis.Table):
            raise TypeError("data must be either a pandas DataFrame or an ibis Table")

        cols = ColumnHelper()

        if isinstance(segment_col, str):
            segment_col = [segment_col]

        required_cols = [
            cols.unit_spend,
            cols.transaction_id,
            *segment_col,
            *filter(lambda x: x in data.columns, [cols.unit_qty, cols.customer_id]),
        ]

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
        self.calc_rollup = calc_rollup
        self.rollup_value = rollup_value

        self.table = self._calc_seg_stats(data, segment_col, calc_total, self.extra_aggs, calc_rollup, rollup_value)

    @staticmethod
    def _get_col_order(include_quantity: bool, include_customer: bool) -> list[str]:
        """Returns the default column order.

        Args:
            include_quantity (bool): Whether to include the columns related to quantity.
            include_customer (bool): Whether to include customer-based columns.

        Returns:
            list[str]: The default column order.
        """
        cols = ColumnHelper()

        column_configs = [
            (cols.agg_unit_spend, True),
            (cols.agg_transaction_id, True),
            (cols.agg_customer_id, include_customer),
            ("units", include_quantity),
            (cols.calc_spend_per_cust, include_customer),
            (cols.calc_spend_per_trans, True),
            (cols.calc_trans_per_cust, include_customer),
            (cols.calc_price_per_unit, include_quantity),
            (cols.calc_units_per_trans, include_quantity),
        ]

        return [col for col, condition in column_configs if condition]

    @staticmethod
    def _create_typed_literals(
        data: ibis.Table,
        columns: list[str],
        values: list[Any],
    ) -> dict[str, ibis.expr.types.generic.Scalar]:
        """Create a dictionary of ibis literals with proper column types.

        Args:
            data (ibis.Table): The data table containing column type information
            columns (list[str]): List of column names
            values (list[Any]): List of values to convert to typed literals

        Returns:
            dict[str, ibis.expr.types.generic.Scalar]: Dictionary mapping column names to typed literals
        """
        mutations = {}
        for i, col in enumerate(columns):
            col_type = data[col].type()
            mutations[col] = ibis.literal(values[i], type=col_type)
        return mutations

    @staticmethod
    def _calc_seg_stats(
        data: ibis.Table,
        segment_col: str | list[str],
        calc_total: bool = True,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
        calc_rollup: bool = False,
        rollup_value: Any | list[Any] = "Total",  # noqa: ANN401 - Any is required for ibis.literal typing
    ) -> ibis.Table:
        """Calculates the transaction statistics by segment.

        Args:
            data (ibis.Table): The transaction data.
            segment_col (list[str]): The columns to use for the segmentation.
            calc_total (bool, optional): Whether to include the total row. Defaults to True.
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations to perform.
                The keys in the dictionary will be the column names for the aggregation results.
                The values are tuples with (column_name, aggregation_function).
            calc_rollup (bool, optional): Whether to calculate rollup totals. Defaults to False.
            rollup_value (Any | list[Any], optional): The value to use for rollup totals. Can be a single value
                applied to all columns or a list of values matching the length of segment_col, with each value
                cast to match the corresponding column type. Defaults to "Total".

        Returns:
            pd.DataFrame: The transaction statistics by segment.

        """
        cols = ColumnHelper()

        # Ensure segment_col is a list
        segment_col = [segment_col] if isinstance(segment_col, str) else segment_col

        # Normalize rollup_value to always be a list matching segment_col length
        rollup_value = [rollup_value] * len(segment_col) if not isinstance(rollup_value, list) else rollup_value

        # Validate rollup_value list length
        if len(rollup_value) != len(segment_col):
            msg = f"If rollup_value is a list, its length must match the number of segment columns. Expected {len(segment_col)}, got {len(rollup_value)}"
            raise ValueError(msg)

        # Base aggregations for segments
        agg_specs = [
            (cols.agg_unit_spend, cols.unit_spend, "sum"),
            (cols.agg_transaction_id, cols.transaction_id, "nunique"),
            (cols.agg_unit_qty, cols.unit_qty, "sum"),
            (cols.agg_customer_id, cols.customer_id, "nunique"),
        ]

        aggs = {agg_name: getattr(data[col], func)() for agg_name, col, func in agg_specs if col in data.columns}

        # Add extra aggregations if provided
        if extra_aggs:
            aggs.update({agg_name: getattr(data[col], func)() for agg_name, (col, func) in extra_aggs.items()})

        # Calculate metrics for segments
        segment_metrics = data.group_by(segment_col).aggregate(**aggs)
        final_metrics = segment_metrics

        # Calculate rollup totals if requested
        if calc_rollup:
            rollup_metrics = []

            # Generate all prefixes of segment_col (except the empty prefix which is the grand total)
            # and excluding the full segment_col list which is already calculated
            for i in range(1, len(segment_col)):
                prefix = segment_col[:i]

                # Group by the prefix and aggregate
                rollup_result = data.group_by(prefix).aggregate(**aggs)

                # Add rollup values for the remaining columns with proper types
                remaining_cols = segment_col[i:]
                remaining_values = rollup_value[i:]
                rollup_mutations = SegTransactionStats._create_typed_literals(data, remaining_cols, remaining_values)

                # Apply all mutations at once
                rollup_result = rollup_result.mutate(**rollup_mutations)
                rollup_metrics.append(rollup_result)

            # Union all rollup results
            if rollup_metrics:
                final_metrics = final_metrics.union(*rollup_metrics)

        # Add grand total row if requested
        if calc_total:
            total_metrics = data.aggregate(**aggs)

            # Create properly typed values for all segment columns
            total_mutations = SegTransactionStats._create_typed_literals(data, segment_col, rollup_value)
            total_metrics = total_metrics.mutate(**total_mutations)
            final_metrics = final_metrics.union(total_metrics)

        # Calculate derived metrics
        final_metrics = final_metrics.mutate(
            **{
                cols.calc_spend_per_trans: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_transaction_id],
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

        if cols.customer_id in data.columns:
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc_spend_per_cust: ibis._[cols.agg_unit_spend] / ibis._[cols.agg_customer_id],
                    cols.calc_trans_per_cust: ibis._[cols.agg_transaction_id]
                    / ibis._[cols.agg_customer_id].cast("float"),
                },
            )

        return final_metrics

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the transaction statistics by segment."""
        if self._df is None:
            cols = ColumnHelper()
            include_quantity = cols.agg_unit_qty in self.table.columns
            include_customer = cols.agg_customer_id in self.table.columns
            col_order = [
                *self.segment_col,
                *SegTransactionStats._get_col_order(
                    include_quantity=include_quantity,
                    include_customer=include_customer,
                ),
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
