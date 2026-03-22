"""New-Repeating-Lapsed Segmentation for Customer Lifecycle Classification.

## Business Context

Understanding customer lifecycle stages is fundamental to retail strategy. Customers naturally
move between periods of activity and inactivity, and identifying these transitions enables
targeted interventions at each stage.

## The Business Problem

Retailers need to understand which customers are growing the base (new), which are loyal
(repeating), and which have stopped purchasing (lapsed). Without this classification,
marketing budgets are misallocated — spending acquisition dollars on existing customers
or ignoring at-risk customers who are about to churn.

## Segment Definitions

Given two time periods (P1 and P2), customers are classified based on where they have
positive spend:

- **New**: Positive spend in P2 only — these customers were acquired in the later period
- **Repeating**: Positive spend in both P1 and P2 — these customers are retained
- **Lapsed**: Positive spend in P1 only — these customers have stopped purchasing

A customer must have positive aggregated spend (> 0) in a period to be considered as having
"bought" in that period. Zero or negative spend does not count.

## Real-World Applications

### New Customers
- Measure acquisition effectiveness period-over-period
- Design onboarding journeys to convert first-time buyers to repeat customers
- Track new customer quality (spend levels, category breadth)

### Repeating Customers
- Core loyal base driving consistent revenue
- Cross-sell and upsell opportunities
- Loyalty program engagement and reward optimization

### Lapsed Customers
- Win-back campaigns with targeted incentives
- Churn root cause analysis (price sensitivity, assortment gaps)
- Customer lifetime value recalculation and reactivation ROI modeling
"""

from __future__ import annotations

import ibis
import pandas as pd

from pyretailscience.options import ColumnHelper

SEGMENT_NEW = "New"
SEGMENT_REPEATING = "Repeating"
SEGMENT_LAPSED = "Lapsed"


class LapseSegmentation:
    """Segments customers into New, Repeating, and Lapsed based on presence across two periods."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str | int,
        p2_value: str | int,
        value_col: str | None = None,
        agg_func: str = "sum",
        group_col: str | list[str] | None = None,
    ) -> None:
        """Segments customers into New, Repeating, and Lapsed based on positive spend across two periods.

        A customer is considered to have "bought" in a period only if their aggregated value_col
        is strictly positive (> 0). Customers are then classified as:
        - New: positive spend in P2 only
        - Repeating: positive spend in both P1 and P2
        - Lapsed: positive spend in P1 only

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data. Must contain customer_id, period_col,
                and value_col columns.
            period_col (str): Column containing period identifiers.
            p1_value (str | int): Value in period_col identifying period 1.
            p2_value (str | int): Value in period_col identifying period 2.
            value_col (str | None): Column to aggregate for determining positive spend.
                Defaults to ColumnHelper().unit_spend.
            agg_func (str): Aggregation function to use when grouping by customer_id.
                Defaults to "sum".
            group_col (str | list[str] | None): Column(s) to group by when calculating segments.
                When specified, segments are calculated within each group independently.
                Defaults to None.

        Raises:
            ValueError: If required columns are missing from the DataFrame, or if p1_value/p2_value
                are not found in period_col.
        """
        self._df: pd.DataFrame | None = None

        cols = ColumnHelper()
        value_col = cols.unit_spend if value_col is None else value_col

        self._group_col: list[str] | None = [group_col] if isinstance(group_col, str) else group_col

        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)

        # Validate required columns
        required_cols = [cols.customer_id, value_col, period_col]
        if self._group_col is not None:
            required_cols.extend(self._group_col)

        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        # Validate p1_value and p2_value exist in period_col
        distinct_periods = df.select(period_col).distinct().execute()[period_col].tolist()
        if p1_value not in distinct_periods:
            msg = f"p1_value '{p1_value}' not found in column '{period_col}'. Available values: {distinct_periods}"
            raise ValueError(msg)
        if p2_value not in distinct_periods:
            msg = f"p2_value '{p2_value}' not found in column '{period_col}'. Available values: {distinct_periods}"
            raise ValueError(msg)

        # Filter to only P1 and P2 rows
        df = df.filter((df[period_col] == p1_value) | (df[period_col] == p2_value))

        # Build group-by columns
        group_by_cols = [cols.customer_id, period_col]
        if self._group_col is not None:
            group_by_cols.extend(self._group_col)

        # Aggregate and filter to positive spend only
        agg_df = df.group_by(*group_by_cols).aggregate(
            **{value_col: getattr(df[value_col], agg_func)()},
        )
        agg_df = agg_df.filter(agg_df[value_col] > 0)

        # Split into P1 and P2 buyer sets with marker columns
        join_cols = [cols.customer_id]
        if self._group_col is not None:
            join_cols.extend(self._group_col)

        select_cols = list(join_cols)
        p1_buyers = (
            agg_df.filter(agg_df[period_col] == p1_value)
            .select(*select_cols)
            .mutate(
                _in_p1=ibis.literal(True),
            )
        )
        p2_buyers = (
            agg_df.filter(agg_df[period_col] == p2_value)
            .select(*select_cols)
            .mutate(
                _in_p2=ibis.literal(True),
            )
        )

        # Outer join to find all customers across both periods
        joined = p1_buyers.outer_join(p2_buyers, join_cols)

        # Coalesce customer_id and group columns from both sides
        coalesce_exprs = {col: ibis.coalesce(joined[f"{col}_right"], joined[col]) for col in join_cols}
        joined = joined.select(**coalesce_exprs, _in_p1=joined["_in_p1"], _in_p2=joined["_in_p2"])

        # Classify: both periods -> Repeating, P1 only -> Lapsed, P2 only -> New
        segment_expr = ibis.cases(
            (joined["_in_p1"].notnull() & joined["_in_p2"].notnull(), SEGMENT_REPEATING),  # noqa: PD004
            (joined["_in_p1"].notnull(), SEGMENT_LAPSED),  # noqa: PD004
            (joined["_in_p2"].notnull(), SEGMENT_NEW),  # noqa: PD004
        )

        result = joined.mutate(segment_name=segment_expr)
        self.table = result.select(*select_cols, "segment_name")

    @property
    def df(self) -> pd.DataFrame:
        """Returns the DataFrame with segment names, indexed by customer_id (and group_col if specified)."""
        if self._df is None:
            cols = ColumnHelper()
            index_cols = [cols.customer_id]
            if self._group_col is not None:
                index_cols.extend(self._group_col)
            self._df = self.table.execute().set_index(index_cols)
        return self._df
