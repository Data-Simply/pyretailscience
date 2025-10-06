"""Segment Performance Analysis for Retail Business Intelligence.

## Business Context

Retailers need to understand performance differences across various business dimensions -
whether comparing customer segments, store locations, product categories, brands, channels,
or any other grouping. This module transforms transactional data into actionable insights
by calculating key performance metrics for any segment or combination of segments.

## The Business Problem

Business stakeholders receive segment data but struggle to answer performance questions:
- Which stores/categories/customer segments generate the most revenue?
- How do transaction patterns differ between segments?
- What's the customer density and spending behavior by segment?
- Are certain combinations of segments more valuable than others?

Without segment performance analysis, decisions are made on incomplete information
rather than data-driven insights about segment value and behavior.

## Real-World Applications

### Customer Segment Analysis
- Compare RFM segments: Which customer types drive the most revenue?
- Analyze geographic segments: Regional performance differences
- Age/demographic segments: Spending patterns by customer characteristics

### Store/Location Analysis
- Store performance comparison: Revenue per customer, transaction frequency
- Regional analysis: Market penetration and customer behavior by area
- Channel analysis: Online vs in-store performance metrics

### Product/Category Analysis
- Category performance: Which product lines drive customer frequency?
- Brand analysis: Private label vs national brand customer behavior
- SKU analysis: Performance metrics for product rationalization decisions

### Multi-Dimensional Analysis
- Store + Customer segment: High-value customers by location
- Category + Channel: Product performance across sales channels
- Brand + Geography: Regional brand performance variations

This module calculates comprehensive statistics including spend, customer counts,
transaction frequency, average basket size, and custom business metrics for any
segment combination.
"""

from typing import Any, Literal

import ibis
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.plots.styles.tailwind import COLORS


class SegTransactionStats:
    """Calculates transaction performance statistics for any business segment or dimension.

    Analyzes transaction data across segments like customer types, store locations,
    product categories, brands, channels, or any combination to reveal performance
    differences and guide business decisions.

    The class automatically calculates key retail metrics including total spend,
    unique customers, transaction frequency, spend per customer, and custom
    aggregations for comparison across segments.
    """

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        data: pd.DataFrame | ibis.Table,
        segment_col: str | list[str] = "segment_name",
        calc_total: bool = True,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
        calc_rollup: bool = False,
        rollup_value: Any | list[Any] = "Total",  # noqa: ANN401 - Any is required for ibis.literal typing
        unknown_customer_value: int | str | ibis.expr.types.Scalar | ibis.expr.types.BooleanColumn | None = None,
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
            calc_rollup (bool, optional): Whether to calculate rollup totals. Defaults to False. When True and
                multiple segment columns are provided, the method generates subtotal rows for both:
                - Prefix rollups: progressively aggregating left-to-right (e.g., [A, B, Total], [A, Total, Total]).
                - Suffix rollups: progressively aggregating right-to-left (e.g., [Total, B, C], [Total, Total, C]).
                A grand total row is also included when calc_total is True.
                Performance: adds O(n) extra aggregation passes where n is the number of segment
                columns. For large hierarchies, consider disabling rollups or reducing columns.
            rollup_value (Any | list[Any], optional): The value to use for rollup totals. Can be a single value
                applied to all columns or a list of values matching the length of segment_col, with each value
                cast to match the corresponding column type. Defaults to "Total".
            unknown_customer_value (int | str | ibis.expr.types.Scalar | ibis.expr.types.BooleanColumn | None, optional):
                Value or expression identifying unknown customers for separate tracking. When provided,
                metrics are split into identified, unknown, and total variants. Accepts simple values (e.g., -1),
                ibis literals, or boolean expressions (e.g., data["customer_id"] < 0). Requires customer_id column.
                Defaults to None.
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
        self.unknown_customer_value = unknown_customer_value

        self.table = self._calc_seg_stats(
            data,
            segment_col,
            calc_total,
            self.extra_aggs,
            calc_rollup,
            rollup_value,
            unknown_customer_value,
        )

    @staticmethod
    def _get_col_order(include_quantity: bool, include_customer: bool, include_unknown: bool = False) -> list[str]:
        """Returns the default column order.

        Args:
            include_quantity (bool): Whether to include the columns related to quantity.
            include_customer (bool): Whether to include customer-based columns.
            include_unknown (bool): Whether to include unknown customer columns. Defaults to False.

        Returns:
            list[str]: The default column order.
        """
        cols = ColumnHelper()

        column_configs = [
            (cols.agg_unit_spend, True),
            (cols.agg_transaction_id, True),
            (cols.agg_customer_id, include_customer),
            (cols.agg_unit_qty, include_quantity),
            (cols.calc_spend_per_cust, include_customer),
            (cols.calc_spend_per_trans, True),
            (cols.calc_trans_per_cust, include_customer),
            (cols.calc_price_per_unit, include_quantity),
            (cols.calc_units_per_trans, include_quantity),
        ]

        # Add unknown customer columns if tracking unknown customers
        if include_unknown:
            unknown_configs = [
                (cols.agg_unit_spend_unknown, True),
                (cols.agg_transaction_id_unknown, True),
                (cols.agg_unit_qty_unknown, include_quantity),
                (cols.calc_spend_per_trans_unknown, True),
                (cols.calc_price_per_unit_unknown, include_quantity),
                (cols.calc_units_per_trans_unknown, include_quantity),
            ]
            column_configs.extend(unknown_configs)

            # Add total columns
            total_configs = [
                (cols.agg_unit_spend_total, True),
                (cols.agg_transaction_id_total, True),
                (cols.agg_unit_qty_total, include_quantity),
                (cols.calc_spend_per_trans_total, True),
                (cols.calc_price_per_unit_total, include_quantity),
                (cols.calc_units_per_trans_total, include_quantity),
            ]
            column_configs.extend(total_configs)

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
    def _add_rollup_metrics(
        data: ibis.Table,
        segment_metrics: ibis.Table,
        segment_col: list[str],
        rollup_value: list[Any],
        aggs: dict[str, Any],
        calc_total: bool,
    ) -> ibis.Table:
        """Add rollup metrics to segment metrics.

        Args:
            data (ibis.Table): The data table
            segment_metrics (ibis.Table): The segment metrics table
            segment_col (list[str]): The segment columns
            rollup_value (list[Any]): The rollup values
            aggs (dict[str, Any]): The aggregation specifications
            calc_total (bool): Whether to include suffix rollups

        Returns:
            ibis.Table: Metrics with rollups added
        """
        rollup_metrics = []

        # Configuration for rollups
        rollup_configs = [
            # Prefix rollups: always include when calc_rollup=True
            (segment_col[:i], segment_col[i:], rollup_value[i:])
            for i in range(1, len(segment_col))
        ]

        # Only add suffix rollups when calc_total=True (to avoid "Total" in category when no grand total)
        if calc_total:
            rollup_configs.extend(
                # Suffix rollups: group by suffixes, mutate preceding columns
                [(segment_col[i:], segment_col[:i], rollup_value[:i]) for i in range(1, len(segment_col))],
            )

        # Process both prefix and suffix rollups with unified logic
        for group_cols, mutation_cols, mutation_values in rollup_configs:
            # Group by the specified columns and aggregate
            rollup_result = data.group_by(group_cols).aggregate(**aggs)

            # Add rollup values for mutation columns with proper types
            rollup_mutations = SegTransactionStats._create_typed_literals(data, mutation_cols, mutation_values)

            # Apply all mutations at once
            rollup_result = rollup_result.mutate(**rollup_mutations)
            rollup_metrics.append(rollup_result)

        # Union all rollup results
        return segment_metrics.union(*rollup_metrics) if rollup_metrics else segment_metrics

    @staticmethod
    def _create_unknown_flag(
        data: ibis.Table,
        unknown_customer_value: int | str | ibis.expr.types.Scalar | ibis.expr.types.BooleanColumn,
    ) -> ibis.expr.types.BooleanColumn:
        """Create a boolean flag identifying unknown customers.

        Args:
            data (ibis.Table): The data table
            unknown_customer_value (int | str | ibis.expr.types.Scalar | ibis.expr.types.BooleanColumn):
                The value or expression identifying unknown customers

        Returns:
            ibis.expr.types.BooleanColumn: Boolean expression identifying unknown customers
        """
        cols = ColumnHelper()

        if isinstance(unknown_customer_value, ibis.expr.types.BooleanColumn):
            return unknown_customer_value
        if isinstance(unknown_customer_value, ibis.expr.types.Scalar):
            return data[cols.customer_id] == unknown_customer_value
        # Simple value (int/str)
        return data[cols.customer_id] == ibis.literal(unknown_customer_value)

    @staticmethod
    def _build_standard_aggs(
        data: ibis.Table,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build standard aggregations without unknown customer tracking.

        Args:
            data (ibis.Table): The data table
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations

        Returns:
            dict[str, Any]: Aggregation specifications
        """
        cols = ColumnHelper()
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

        return aggs

    @staticmethod
    def _build_unknown_aggs(
        data: ibis.Table,
        unknown_flag: ibis.expr.types.BooleanColumn,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Build aggregations with unknown customer tracking.

        Args:
            data (ibis.Table): The data table
            unknown_flag (ibis.expr.types.BooleanColumn): Boolean flag identifying unknown customers
            extra_aggs (dict[str, tuple[str, str]] | None): Additional aggregations

        Returns:
            dict[str, Any]: Aggregation specifications for identified, unknown, and total variants
        """
        cols = ColumnHelper()
        aggs = {}

        # Identified customers only (where NOT unknown)
        # Use coalesce to ensure proper types: int for counts, float for sums
        aggs[cols.agg_unit_spend] = data[cols.unit_spend].sum(where=~unknown_flag).coalesce(0.0)
        aggs[cols.agg_transaction_id] = data[cols.transaction_id].nunique(where=~unknown_flag).coalesce(0)
        aggs[cols.agg_customer_id] = data[cols.customer_id].nunique(where=~unknown_flag).coalesce(0)
        if cols.unit_qty in data.columns:
            aggs[cols.agg_unit_qty] = data[cols.unit_qty].sum(where=~unknown_flag).coalesce(0)

        # Unknown customers (where unknown)
        # Use coalesce to ensure proper types: int for counts, float for sums
        aggs[cols.agg_unit_spend_unknown] = data[cols.unit_spend].sum(where=unknown_flag).coalesce(0.0)
        aggs[cols.agg_transaction_id_unknown] = data[cols.transaction_id].nunique(where=unknown_flag).coalesce(0)
        if cols.unit_qty in data.columns:
            aggs[cols.agg_unit_qty_unknown] = data[cols.unit_qty].sum(where=unknown_flag).coalesce(0)

        # Total (all customers)
        aggs[cols.agg_unit_spend_total] = data[cols.unit_spend].sum()
        aggs[cols.agg_transaction_id_total] = data[cols.transaction_id].nunique()
        if cols.unit_qty in data.columns:
            aggs[cols.agg_unit_qty_total] = data[cols.unit_qty].sum()

        # Add extra aggregations with three variants
        if extra_aggs:
            suffix_unknown = get_option("column.suffix.unknown_customer")
            suffix_total = get_option("column.suffix.total")
            for agg_name, (col, func) in extra_aggs.items():
                # Use coalesce with 0 for count functions, 0.0 for others
                coalesce_value = 0 if func in ("nunique", "count") else 0.0
                aggs[agg_name] = getattr(data[col], func)(where=~unknown_flag).coalesce(coalesce_value)
                aggs[f"{agg_name}_{suffix_unknown}"] = getattr(data[col], func)(where=unknown_flag).coalesce(
                    coalesce_value,
                )
                aggs[f"{agg_name}_{suffix_total}"] = getattr(data[col], func)()

        return aggs

    @staticmethod
    def _calc_seg_stats(
        data: ibis.Table,
        segment_col: str | list[str],
        calc_total: bool = True,
        extra_aggs: dict[str, tuple[str, str]] | None = None,
        calc_rollup: bool = False,
        rollup_value: Any | list[Any] = "Total",  # noqa: ANN401 - Any is required for ibis.literal typing
        unknown_customer_value: int | str | ibis.expr.types.Scalar | ibis.expr.types.BooleanColumn | None = None,
    ) -> ibis.Table:
        """Calculates the transaction statistics by segment.

        Args:
            data (ibis.Table): The transaction data.
            segment_col (list[str]): The columns to use for the segmentation.
            calc_total (bool, optional): Whether to include the total row. Defaults to True.
            extra_aggs (dict[str, tuple[str, str]], optional): Additional aggregations to perform.
                The keys in the dictionary will be the column names for the aggregation results.
                The values are tuples with (column_name, aggregation_function).
            calc_rollup (bool, optional): Whether to calculate rollup totals. Defaults to False. When True with
                multiple segment columns, subtotal rows are added for all non-empty prefixes and suffixes of the
                hierarchy. For example, with [A, B, C], prefixes include [A, B, Total], [A, Total, Total]; suffixes
                include [Total, B, C], [Total, Total, C]. Performance: O(n) additional aggregation passes for suffixes,
                where n is the number of segment columns.
            rollup_value (Any | list[Any], optional): The value to use for rollup totals. Can be a single value
                applied to all columns or a list of values matching the length of segment_col, with each value
                cast to match the corresponding column type. Defaults to "Total".
            unknown_customer_value (int | str | ibis.expr.types.Scalar | ibis.expr.types.BooleanColumn | None, optional):
                Value or expression identifying unknown customers for separate tracking. When provided,
                metrics are split into identified, unknown, and total variants. Accepts simple values (e.g., -1),
                ibis literals, or boolean expressions. Defaults to None.

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

        # Validate and create unknown flag if unknown_customer_value is provided
        unknown_flag = None
        if unknown_customer_value is not None:
            if cols.customer_id not in data.columns:
                msg = f"Column '{cols.customer_id}' is required when unknown_customer_value parameter is specified"
                raise ValueError(msg)
            unknown_flag = SegTransactionStats._create_unknown_flag(data, unknown_customer_value)

        # Build aggregations based on unknown customer tracking
        aggs = (
            SegTransactionStats._build_unknown_aggs(data, unknown_flag, extra_aggs)
            if unknown_flag is not None
            else SegTransactionStats._build_standard_aggs(data, extra_aggs)
        )

        # Calculate metrics for segments
        segment_metrics = data.group_by(segment_col).aggregate(**aggs)

        # Add rollups if requested
        final_metrics = (
            SegTransactionStats._add_rollup_metrics(data, segment_metrics, segment_col, rollup_value, aggs, calc_total)
            if calc_rollup
            else segment_metrics
        )

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

        # Add derived metrics for unknown and total when tracking unknown customers
        if unknown_flag is not None:
            # Unknown customer derived metrics
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc_spend_per_trans_unknown: ibis._[cols.agg_unit_spend_unknown]
                    / ibis._[cols.agg_transaction_id_unknown],
                },
            )

            # Total derived metrics
            final_metrics = final_metrics.mutate(
                **{
                    cols.calc_spend_per_trans_total: ibis._[cols.agg_unit_spend_total]
                    / ibis._[cols.agg_transaction_id_total],
                },
            )

            # Quantity-based derived metrics for unknown and total
            if cols.unit_qty in data.columns:
                final_metrics = final_metrics.mutate(
                    **{
                        cols.calc_price_per_unit_unknown: ibis._[cols.agg_unit_spend_unknown]
                        / ibis._[cols.agg_unit_qty_unknown].nullif(0),
                        cols.calc_units_per_trans_unknown: ibis._[cols.agg_unit_qty_unknown]
                        / ibis._[cols.agg_transaction_id_unknown].cast("float"),
                        cols.calc_price_per_unit_total: ibis._[cols.agg_unit_spend_total]
                        / ibis._[cols.agg_unit_qty_total].nullif(0),
                        cols.calc_units_per_trans_total: ibis._[cols.agg_unit_qty_total]
                        / ibis._[cols.agg_transaction_id_total].cast("float"),
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
            include_unknown = self.unknown_customer_value is not None
            col_order = [
                *self.segment_col,
                *SegTransactionStats._get_col_order(
                    include_quantity=include_quantity,
                    include_customer=include_customer,
                    include_unknown=include_unknown,
                ),
            ]

            # Add any extra aggregation columns to the column order
            if hasattr(self, "extra_aggs") and self.extra_aggs:
                if include_unknown:
                    # Add identified, unknown, and total variants for each extra agg
                    suffix_unknown = get_option("column.suffix.unknown_customer")
                    suffix_total = get_option("column.suffix.total")
                    for agg_name in self.extra_aggs:
                        col_order.append(agg_name)
                        col_order.append(f"{agg_name}_{suffix_unknown}")
                        col_order.append(f"{agg_name}_{suffix_total}")
                else:
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
