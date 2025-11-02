"""Utility functions for time-related operations in retail analysis."""

from collections.abc import Mapping
from datetime import UTC, datetime

import ibis
import numpy as np
import pandas as pd

from pyretailscience.options import get_option


def _normalize_datetime(date_val: datetime | str) -> datetime:
    """Convert string or datetime to timezone-aware datetime object."""
    if isinstance(date_val, str):
        # Convert string to timezone-aware datetime
        return datetime.strptime(date_val, "%Y-%m-%d").replace(tzinfo=UTC)
    if isinstance(date_val, datetime):
        # If datetime is timezone-naive, make it timezone-aware (UTC)
        if date_val.tzinfo is None:
            return date_val.replace(tzinfo=UTC)
        return date_val
    error_msg = f"Expected str or datetime, got {type(date_val)}"
    raise TypeError(error_msg)


def _validate_and_normalize_periods(
    period_ranges: Mapping[str, tuple[datetime | str, datetime | str]],
) -> dict[str, tuple[datetime, datetime]]:
    """Validates and normalizes period ranges, returning timezone-aware datetime tuples."""
    normalized = {}

    for period_name, date_range in period_ranges.items():
        date_range_length = 2
        if not (isinstance(date_range, tuple) and len(date_range) == date_range_length):
            msg = f"Period '{period_name}' must have a (start_date, end_date) tuple"
            raise ValueError(msg)

        start_date, end_date = date_range

        start_dt = _normalize_datetime(start_date)
        end_dt = _normalize_datetime(end_date)

        if start_dt > end_dt:
            msg = f"Period '{period_name}': start date ({start_date}) must be <= end date ({end_date})"
            raise ValueError(msg)

        normalized[period_name] = (start_dt, end_dt)

    # Check for overlapping periods
    period_list = list(normalized.items())
    for i, (name1, (start1, end1)) in enumerate(period_list):
        for name2, (start2, end2) in period_list[i + 1 :]:
            if start1 <= end2 and start2 <= end1:
                overlap_msg = f"Periods '{name1}' ({start1.date()}-{end1.date()}) and '{name2}' ({start2.date()}-{end2.date()}) overlap"
                raise ValueError(overlap_msg)

    return normalized


def filter_and_label_by_periods(
    transactions: ibis.Table,
    period_ranges: dict[str, tuple[datetime, datetime] | tuple[str, str]],
    period_col: str = "period_name",
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
        period_col (str): Name of the column to create for period labels. Defaults to "period_name".

    Returns:
        An ibis table with filtered transactions and added period label column.

    Raises:
        ValueError: If any value in period_ranges is not a tuple of length 2.
        ValueError: If first date > second date for any period.
        ValueError: If periods overlap with each other.
    """
    # Validate periods first
    _validate_and_normalize_periods(period_ranges)

    branches = []
    date_column = transactions[get_option("column.transaction_date")]

    date_col_dtype = date_column.type()

    for period_name, date_range in period_ranges.items():
        start_date, end_date = date_range

        period_condition = date_column.between(
            ibis.literal(start_date, type=date_col_dtype),
            ibis.literal(end_date, type=date_col_dtype),
        )
        branches.append((period_condition, ibis.literal(period_name)))

    conditions = ibis.or_(*[condition[0] for condition in branches])
    return transactions.filter(conditions).mutate(**{period_col: ibis.cases(*branches)})


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
