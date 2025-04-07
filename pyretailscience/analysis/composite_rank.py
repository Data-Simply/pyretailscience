"""Composite Rank Analysis Module.

This module provides the `CompositeRank` class which creates a composite ranking of
several columns by giving each column an individual rank and then combining those
ranks together into a single composite rank.

Key Features:
- Creates individual ranks for multiple columns
- Supports both ascending and descending sort orders for each column
- Combines individual ranks using a specified aggregation function
- Can handle tie values with configurable options
- Utilizes Ibis for efficient query execution
"""

import ibis
import pandas as pd


class CompositeRank:
    """Creates a composite rank from multiple columns.

    This class creates a composite rank of several columns by giving each column an
    individual rank, and then combining those ranks together into a single composite rank.
    Composite ranks are often used in product range reviews when there are several important
    factors to consider when deciding to list or delist a product.
    """

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        rank_cols: list[tuple[str, str] | str],
        agg_func: str,
        ignore_ties: bool = False,
    ) -> None:
        """Initialize the CompositeRank class.

        Args:
            df (pd.DataFrame | ibis.Table): An ibis table or pandas DataFrame containing the data.
            rank_cols (List[Union[Tuple[str, str], str]]): A list of columns to create the composite rank on.
                Can be specified as tuples of (column_name, sort_order) where sort_order is 'asc', 'ascending',
                'desc', or 'descending'. If just a string is provided, ascending order is assumed.
            agg_func (str): The aggregation function to use when combining ranks.
                Supported values are "mean", "sum", "min", "max".
            ignore_ties (bool, optional): Whether to ignore ties when calculating ranks. If True, will use
                row_number (each row gets a unique rank). If False (default), will use rank (ties get the same rank).

        Raises:
            ValueError: If any of the specified columns are not in the DataFrame or if a sort order is invalid.
            ValueError: If the aggregation function is not one of the supported values.
        """
        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)

        # Validate columns and sort orders
        valid_sort_orders = ["asc", "ascending", "desc", "descending"]

        rank_mutates = {}
        for col_spec in rank_cols:
            if isinstance(col_spec, str):
                col_name = col_spec
                sort_order = "asc"
            else:
                if len(col_spec) != 2:  # noqa: PLR2004 - Error message below explains the value
                    msg = (
                        f"Column specification must be a string or a tuple of (column_name, sort_order). Got {col_spec}"
                    )
                    raise ValueError(msg)
                col_name, sort_order = col_spec

            if col_name not in df.columns:
                msg = f"Column '{col_name}' not found in the DataFrame."
                raise ValueError(msg)

            if sort_order.lower() not in valid_sort_orders:
                msg = f"Sort order must be one of {valid_sort_orders}. Got '{sort_order}'"
                raise ValueError(msg)

            order_by = ibis.asc(df[col_name]) if sort_order in ["asc", "ascending"] else ibis.desc(df[col_name])
            window = ibis.window(order_by=order_by)

            # Calculate rank based on ignore_ties parameter (using 1-based ranks)
            # ibis.row_number() is 1-based, ibis.rank() is 0-based so we add 1
            rank_col = ibis.row_number().over(window) if ignore_ties else ibis.rank().over(window) + 1

            # Add the rank column to the result table
            rank_mutates[f"{col_name}_rank"] = rank_col

        df = df.mutate(**rank_mutates)

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

        self.table = df.mutate(composite_rank=agg_expr[agg_func])

    @property
    def df(self) -> pd.DataFrame:
        """Returns the pandas DataFrame representation of the table.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the original data,
                individual column ranks, and the composite rank.
        """
        if self._df is None:
            self._df = self.table.execute()
        return self._df
