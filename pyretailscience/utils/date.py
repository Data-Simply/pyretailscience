"""Utility functions for time-related operations in retail analysis."""

from datetime import datetime

import ibis

from pyretailscience.options import get_option


def filter_and_label_by_periods(
    transactions: ibis.Table,
    period_ranges: dict[str, tuple[datetime, datetime] | tuple[str, str]],
) -> ibis.Table:
    """Filters transactions to specified time periods and adds period labels.

    This function filters transactions based on specified time periods and adds a new column indicating the period name.
    It is useful for analyzing transactions within specific date ranges and comparing KPIs between them.

    Example:
        transactions = ibis.table("transactions")
        period_ranges = {
            "Q1": ("2023-01-01", "2023-03-31"),
            "Q2": ("2023-04-01", "2023-06-30"),
        }
        filtered_transactions = filter_and_label_by_periods(transactions, period_ranges)
        # filtered_transactions will only contain transactions from the date ranges specified in Q1 and Q2 and a new
        # column 'period_name' will be in the table defining the period for each transaction.

    Args:
        transactions (ibis.Table): An ibis table with a transaction_date column.
        period_ranges (dict[str, tuple[datetime, datetime] | tuple[str, str]]): Dict where keys are period names and
            values are(start_date, end_date) tuples.

    Returns:
        An ibis table with filtered transactions and added period_name column.

    Raises:
        ValueError: If any value in period_ranges is not a tuple of length 2.
    """
    branches = []

    for period_name, date_range in period_ranges.items():
        if not (isinstance(date_range, tuple) and len(date_range) == 2):  # noqa: PLR2004 - Explained in the error below
            msg = f"Period '{period_name}' must have a (start_date, end_date) tuple"
            raise ValueError(msg)

        period_condition = transactions[get_option("column.transaction_date")].between(date_range[0], date_range[1])
        branches.append((period_condition, ibis.literal(period_name)))

    conditions = ibis.or_(*[condition[0] for condition in branches])
    return transactions.filter(conditions).mutate(period_name=ibis.cases(*branches))
