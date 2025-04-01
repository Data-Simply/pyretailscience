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


def find_overlapping_periods(start_date: datetime, end_date: datetime) -> list[tuple[str, str]]:
    """Find overlapping time periods within the given date range, split by year.

    This function generates overlapping periods between a given start date and end date.
    The first period will start from the given start date, and each subsequent period will start on
    the same month and day for the following years, ending each period at the year's end (December 31)
    except for the last period, which ends at the provided end date.

    Args:
        start_date (datetime): The starting date of the range.
        end_date (datetime): The ending date of the range.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the start and end dates of an overlapping period
                               in 'YYYY-MM-DD' format.

    Raises:
        ValueError: If the start date is after the end date.
    """
    if start_date > end_date:
        raise ValueError("Start date must be before end date")

    start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
    end_year = end_date.year

    years = np.arange(start_year, end_year + 1)

    period_starts = [start_date if year == start_year else datetime(year, start_month, start_day) for year in years]  # noqa: DTZ001
    period_ends = [end_date if year == end_year else datetime(year, 12, 31) for year in years]  # noqa: DTZ001

    df = pd.DataFrame({"start": period_starts, "end": period_ends})
    df = df[df["start"] <= df["end"]]

    return [
        (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        for start, end in zip(df["start"], df["end"], strict=False)
    ]
