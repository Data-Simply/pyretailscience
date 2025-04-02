"""Module: Date Utilities.

This module provides utilities for working with date ranges and finding overlapping periods. The main function,
`find_overlapping_periods`, takes a start and end date and returns a list of overlapping periods within the given range,
splitting the range by year.

Functionality:
- Find all overlapping periods between the given start and end date.
- Split the range into periods that start from the provided start date for the first period
  and then yearly thereafter, ending on the given end date.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def find_overlapping_periods(
    start_date: datetime | str,
    end_date: datetime | str,
    return_iso: bool = True,
) -> list[tuple[str | datetime, str | datetime]]:
    """Find overlapping time periods within the given date range, split by year.

    This function generates overlapping periods between a given start date and end date.
    The first period will start from the given start date, and each subsequent period will start on
    he same month and day for the following years, ending each period on the same month and day
    of the end date but in the subsequent year,
    except for the last period, which ends at the provided end date.

    Args:
        start_date (Union[datetime, str]): The starting date of the range, either as a datetime object or 'YYYY-MM-DD' string.
        end_date (Union[datetime, str]): The ending date of the range, either as a datetime object or 'YYYY-MM-DD' string.
        return_iso (bool, optional): If True, returns dates as ISO-formatted strings ('YYYY-MM-DD').
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

    if return_iso:
        return [
            (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            for start, end in zip(df["start"], df["end"], strict=False)
        ]
    return list(zip(df["start"], df["end"], strict=False))
