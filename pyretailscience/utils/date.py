"""Utility functions for time-related operations in retail analysis."""

from datetime import datetime

import ibis
import numpy as np
import pandas as pd

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


def find_overlapping_periods(
    start_date: datetime | str,
    end_date: datetime | str,
    return_str: bool = True,
) -> list[tuple[str | datetime, str | datetime]]:
    """Find overlapping time periods within the given date range, split by year.

    This function generates overlapping periods between a given start date and end date.
    The first period will start from the given start date, and each subsequent period will start on
    the same month and day for the following years, ending each period on the same month and day
    of the end date but in the subsequent year,
    except for the last period, which ends at the provided end date.

    Note:
        This function does not adjust for leap years. If the start or end date is February 29,
        it may cause an issue in non-leap years.

    Args:
        start_date (Union[datetime, str]): The starting date of the range, either as a datetime object or 'YYYY-MM-DD' string.
        end_date (Union[datetime, str]): The ending date of the range, either as a datetime object or 'YYYY-MM-DD' string.
        return_str (bool, optional): If True, returns dates as ISO-formatted strings ('YYYY-MM-DD').
                                     If False, returns datetime objects. Defaults to True.

    Returns:
        List[Tuple[Union[str, datetime], Union[str, datetime]]]:
        A list of tuples where each tuple contains the start and end dates of an overlapping period,
        either as strings (ISO format) or datetime objects.

    Raises:
        ValueError: If the start date is after the end date.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")  # noqa: DTZ007

    if start_date > end_date:
        raise ValueError("Start date must be before end date")

    start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
    end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
    if start_year == end_year:
        return []

    years = np.arange(start_year, end_year)

    period_starts = [start_date if year == start_year else datetime(year, start_month, start_day) for year in years]  # noqa: DTZ001
    period_ends = [datetime(year + 1, end_month, end_day) for year in years]  # noqa: DTZ001

    df = pd.DataFrame({"start": period_starts, "end": period_ends})

    if return_str:
        return [
            (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            for start, end in zip(df["start"], df["end"], strict=False)
        ]
    return list(zip(df["start"], df["end"], strict=False))
