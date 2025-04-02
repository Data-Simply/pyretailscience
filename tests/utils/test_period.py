"""Tests for the Period Overlapping module."""

# flake8: noqa: DTZ001
from datetime import datetime

import pytest

from pyretailscience.utils.period import find_overlapping_periods


class TestFindOverlappingPeriods:
    """Test cases for the find_overlapping_periods function."""

    def test_valid_range_same_year(self):
        """Test case where the start and end dates are in the same year."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_multiple_years(self):
        """Test case where the start and end dates span multiple years."""
        start_date = datetime(2022, 6, 15)
        end_date = datetime(2025, 3, 10)
        expected = [
            ("2022-06-15", "2023-03-10"),
            ("2023-06-15", "2024-03-10"),
            ("2024-06-15", "2025-03-10"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_full_years(self):
        """Test case with a date range spanning full years."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        expected = [
            ("2020-01-01", "2021-12-31"),
            ("2021-01-01", "2022-12-31"),
            ("2022-01-01", "2023-12-31"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_invalid_date_range(self):
        """Test case where the start date is after the end date."""
        start_date = datetime(2024, 12, 31)
        end_date = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="Start date must be before end date"):
            find_overlapping_periods(start_date, end_date)

    def test_same_start_and_end_date(self):
        """Test case where the start and end dates are the same."""
        start_date = datetime(2024, 6, 1)
        end_date = datetime(2024, 6, 1)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_large_date_range(self):
        """Test case with a very large date range spanning many years."""
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2025, 12, 31)
        expected = [(f"{year}-01-01", f"{year + 1}-12-31") for year in range(2000, 2025)]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_multiple_years_datetime(self):
        """Test case where the start and end dates span multiple years with return_iso=False."""
        start_date = datetime(2021, 5, 10)
        end_date = datetime(2024, 8, 20)
        expected = [
            (datetime(2021, 5, 10), datetime(2022, 8, 20)),
            (datetime(2022, 5, 10), datetime(2023, 8, 20)),
            (datetime(2023, 5, 10), datetime(2024, 8, 20)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_iso=False)
        assert result == expected

    def test_valid_range_multiple_years_string(self):
        """Test case where the start and end dates are provided as strings with return_iso=False."""
        start_date = "2021-05-10"
        end_date = "2024-08-20"
        expected = [
            (datetime(2021, 5, 10), datetime(2022, 8, 20)),
            (datetime(2022, 5, 10), datetime(2023, 8, 20)),
            (datetime(2023, 5, 10), datetime(2024, 8, 20)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_iso=False)
        assert result == expected
