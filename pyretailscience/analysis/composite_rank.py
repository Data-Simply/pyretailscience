"""Composite Rank Analysis Module for Multi-Factor Retail Decision Making.

## Business Context

In retail, critical decisions like product ranging, supplier selection, and store
performance evaluation require balancing multiple competing factors. A product might
have high sales but low margin, or a supplier might offer great prices but poor
delivery reliability. Composite ranking enables data-driven decisions by combining
multiple performance metrics into a single, actionable score.

## Real-World Applications

1. **Product Range Optimization**: Rank products for listing/delisting decisions based on:
   - Sales velocity (units per week)
   - Gross margin percentage
   - Stock turn rate
   - Customer satisfaction scores
   - Return rates

2. **Supplier Performance Management**: Evaluate suppliers using:
   - On-time delivery percentage
   - Price competitiveness
   - Quality scores (defect rates)
   - Payment terms flexibility
   - Order fill rates

3. **Store Performance Assessment**: Rank stores for investment decisions based on:
   - Sales per square foot
   - Conversion rates
   - Labor productivity
   - Customer satisfaction (NPS)
   - Shrinkage rates

4. **Category Management**: Prioritize categories for space allocation using:
   - Category growth rates
   - Market share
   - Profitability
   - Cross-category purchase influence
   - Seasonal consistency

## How It Works

The module creates individual rankings for each metric, then combines these rankings
using aggregation functions (mean, sum, min, max) to produce a final composite score.
This approach normalizes metrics with different scales and ensures each factor contributes
appropriately to the final decision.

## Business Value

- **Objective Decision Making**: Removes bias by systematically weighing all factors
- **Scalability**: Can evaluate thousands of products/stores/suppliers simultaneously
- **Transparency**: Clear methodology that stakeholders can understand and trust
- **Flexibility**: Different aggregation methods suit different business strategies
- **Actionable Output**: Direct ranking enables clear cut-off decisions

Key Features:
- Creates individual ranks for multiple columns with business metrics
- Supports both ascending and descending sort orders for each metric
- Combines individual ranks using business-appropriate aggregation functions
- Handles tie values for fair comparison
- Utilizes Ibis for efficient query execution on large retail datasets
"""

import ibis
import pandas as pd


class CompositeRank:
    """Creates multi-factor composite rankings for retail decision-making.

    The CompositeRank class enables retailers to make data-driven decisions by combining
    multiple performance metrics into a single, actionable ranking. This is essential for
    scenarios where no single metric tells the complete story.

    ## Business Problem Solved

    Retailers face complex trade-offs daily: Should we keep the high-volume product with
    low margins or the high-margin product with slow sales? Which supplier offers the best
    overall value when considering price, quality, and reliability? This class provides a
    systematic approach to these multi-dimensional decisions.

    ## Example Use Case: Product Range Review

    When conducting quarterly range reviews, a retailer might rank products by:
    - Sales performance (higher is better → descending order)
    - Days of inventory (lower is better → ascending order)
    - Customer rating (higher is better → descending order)
    - Return rate (lower is better → ascending order)

    The composite rank identifies products that perform well across ALL metrics, not just
    excel in one area. Products with the best composite scores are clear "keep" decisions,
    while those with the worst scores are candidates for delisting.

    ## Aggregation Strategies

    Different business contexts require different aggregation approaches:
    - **Mean**: Balanced scorecard approach, all factors equally important
    - **Min**: Conservative approach, focus on worst-performing metric
    - **Max**: Optimistic approach, highlight strength in any area
    - **Sum**: Cumulative performance across all dimensions

    ## Actionable Outcomes

    The composite rank directly supports decisions like:
    - Top 20% composite rank → Increase inventory investment
    - Bottom 20% composite rank → Consider delisting or markdown
    - Middle 60% → Maintain current strategy, monitor for changes
    """

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        rank_cols: list[tuple[str, str] | str],
        agg_func: str,
        ignore_ties: bool = False,
        group_col: str | list[str] | list | None = None,
    ) -> None:
        """Initialize the CompositeRank class for multi-criteria retail analysis.

        Args:
            df (pd.DataFrame | ibis.Table): Product, store, or supplier performance data.
            rank_cols (List[Union[Tuple[str, str], str]]): Metrics to rank with their optimization direction.
                Examples for product ranging:
                - ("sales_units", "desc") - Higher sales are better
                - ("days_inventory", "asc") - Lower inventory days are better
                - ("margin_pct", "desc") - Higher margins are better
                - ("return_rate", "asc") - Lower returns are better
                If just a string is provided, ascending order is assumed.
            agg_func (str): How to combine individual rankings:
                - "mean": Balanced scorecard (most common for range reviews)
                - "sum": Total performance score (for bonus calculations)
                - "min": Worst-case performance (for risk assessment)
                - "max": Best-case performance (for opportunity identification)
            ignore_ties (bool, optional): How to handle identical values:
                - False (default): Products with same sales get same rank (fair comparison)
                - True: Force unique ranks even for ties (strict ordering needed)
            group_col (str | list[str] | list, optional): Column(s) to partition rankings by group.
                - None (default): Rank across entire dataset (current behavior)
                - If specified: Calculate ranks independently within each group
                Examples for group-based ranking:
                - "product_category": Rank products within each category
                - "store_region": Rank stores within their regions
                - "supplier_type": Rank suppliers within their specialization

        Raises:
            ValueError: If specified metrics are not in the data or sort order is invalid.
            ValueError: If aggregation function is not supported.
            ValueError: If group_col is specified but doesn't exist in the data.

        Examples:
            >>> # Global ranking: Rank all products together (current behavior)
            >>> ranker = CompositeRank(
            ...     df=product_data,
            ...     rank_cols=[
            ...         ("weekly_sales", "desc"),
            ...         ("margin_percentage", "desc"),
            ...         ("stock_cover_days", "asc"),
            ...         ("customer_rating", "desc")
            ...     ],
            ...     agg_func="mean"
            ... )
            >>> # Products with lowest composite_rank should be reviewed for delisting

            >>> # Group-based ranking: Rank products within each category
            >>> ranker = CompositeRank(
            ...     df=product_data,
            ...     rank_cols=[
            ...         ("weekly_sales", "desc"),
            ...         ("margin_percentage", "desc"),
            ...         ("stock_cover_days", "asc")
            ...     ],
            ...     agg_func="mean",
            ...     group_col="product_category"
            ... )
            >>> # Electronics products ranked against other electronics
            >>> # Apparel products ranked against other apparel
        """
        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)

        self._validate_group_col(group_col, df)
        rank_mutates = self._process_rank_columns(rank_cols, df, group_col, ignore_ties)
        df = df.mutate(**rank_mutates)
        self.table = self._create_composite_ranking(df, rank_mutates, agg_func)

    def _validate_group_col(self, group_col: str | list[str] | list | None, df: ibis.Table) -> None:
        """Validate group_col parameter."""
        if group_col is not None:
            if isinstance(group_col, str):
                if group_col not in df.columns:
                    msg = f"Group column '{group_col}' not found in the DataFrame."
                    raise ValueError(msg)
            elif isinstance(group_col, list):
                # Validate that all items in the list are strings
                if not all(isinstance(col, str) for col in group_col):
                    msg = f"All group columns must be strings. Got {group_col}"
                    raise ValueError(msg)
                missing_cols = [col for col in group_col if col not in df.columns]
                if missing_cols:
                    msg = f"Group columns {missing_cols} not found in the DataFrame."
                    raise ValueError(msg)
            else:
                # Invalid type - maintain backward compatibility with error message
                msg = f"Group column '{group_col}' not found in the DataFrame."
                raise ValueError(msg)

    def _process_rank_columns(
        self,
        rank_cols: list[tuple[str, str] | str],
        df: ibis.Table,
        group_col: str | list[str] | list | None,
        ignore_ties: bool,
    ) -> dict[str, any]:
        """Process rank columns and create ranking expressions."""
        valid_sort_orders = ["asc", "ascending", "desc", "descending"]
        rank_mutates = {}

        for col_spec in rank_cols:
            col_name, sort_order = self._parse_column_spec(col_spec)
            self._validate_column_and_sort_order(col_name, sort_order, df, valid_sort_orders)

            order_by = ibis.asc(df[col_name]) if sort_order in ["asc", "ascending"] else ibis.desc(df[col_name])
            window = self._create_window(group_col, df, order_by)

            # Calculate rank based on ignore_ties parameter (using 1-based ranks)
            rank_col = ibis.row_number().over(window) + 1 if ignore_ties else ibis.rank().over(window) + 1
            rank_mutates[f"{col_name}_rank"] = rank_col

        return rank_mutates

    def _parse_column_spec(self, col_spec: tuple[str, str] | str) -> tuple[str, str]:
        """Parse column specification into column name and sort order."""
        if isinstance(col_spec, str):
            return col_spec, "asc"
        if len(col_spec) != 2:  # noqa: PLR2004 - Error message below explains the value
            msg = f"Column specification must be a string or a tuple of (column_name, sort_order). Got {col_spec}"
            raise ValueError(msg)
        return col_spec

    def _validate_column_and_sort_order(
        self,
        col_name: str,
        sort_order: str,
        df: ibis.Table,
        valid_sort_orders: list[str],
    ) -> None:
        """Validate column exists and sort order is valid."""
        if col_name not in df.columns:
            msg = f"Column '{col_name}' not found in the DataFrame."
            raise ValueError(msg)
        if sort_order.lower() not in valid_sort_orders:
            msg = f"Sort order must be one of {valid_sort_orders}. Got '{sort_order}'"
            raise ValueError(msg)

    def _create_window(
        self,
        group_col: str | list[str] | list | None,
        df: ibis.Table,
        order_by: any,
    ) -> any:
        """Create window function for ranking."""
        return (
            ibis.window(order_by=order_by)
            if group_col is None
            else ibis.window(group_by=df[group_col], order_by=order_by)
            if isinstance(group_col, str)
            else ibis.window(group_by=[df[col] for col in group_col], order_by=order_by)
        )

    def _create_composite_ranking(self, df: ibis.Table, rank_mutates: dict[str, any], agg_func: str) -> ibis.Table:
        """Create the final composite ranking."""
        column_refs = [df[col] for col in rank_mutates]
        agg_expr = {
            "mean": sum(column_refs) / len(column_refs),
            "sum": sum(column_refs),
            "min": ibis.least(*column_refs),
            "max": ibis.greatest(*column_refs),
        }

        if agg_func.lower() not in agg_expr:
            msg = f"Aggregation function must be one of {list(agg_expr.keys())}. Got '{agg_func}'"
            raise ValueError(msg)

        return df.mutate(composite_rank=agg_expr[agg_func])

    @property
    def df(self) -> pd.DataFrame:
        """Returns ranked data ready for business decision-making.

        Returns:
            pd.DataFrame: Performance data with ranking columns added:
                - Original metrics (sales, margin, etc.)
                - Individual rank columns (e.g., sales_rank, margin_rank)
                - composite_rank: Final combined ranking for decisions
        """
        if self._df is None:
            self._df = self.table.execute()
        return self._df
