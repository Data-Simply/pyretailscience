"""Tests for the Period Overlapping module."""

from datetime import datetime

import pytest

from pyretailscience.date_utils import find_overlapping_periods


class TestFindOverlappingPeriods:
    """Test cases for the find_overlapping_periods function."""

    def test_valid_range_same_year(self):
        """Test case where the start and end dates are in the same year."""
        start_date = datetime(2024, 1, 1)  # noqa: DTZ001
        end_date = datetime(2024, 12, 31)  # noqa: DTZ001
        expected = [("2024-01-01", "2024-12-31")]
        assert find_overlapping_periods(start_date, end_date) == expected

    def test_valid_range_multiple_years(self):
        """Test case where the start and end dates span multiple years."""
        start_date = datetime(2022, 6, 15)  # noqa: DTZ001
        end_date = datetime(2025, 3, 10)  # noqa: DTZ001
        expected = [
            ("2022-06-15", "2022-12-31"),
            ("2023-06-15", "2023-12-31"),
            ("2024-06-15", "2024-12-31"),
        ]
        assert find_overlapping_periods(start_date, end_date) == expected

    def test_valid_range_full_years(self):
        """Test case with a date range spanning full years."""
        start_date = datetime(2020, 1, 1)  # noqa: DTZ001
        end_date = datetime(2023, 12, 31)  # noqa: DTZ001
        expected = [
            ("2020-01-01", "2020-12-31"),
            ("2021-01-01", "2021-12-31"),
            ("2022-01-01", "2022-12-31"),
            ("2023-01-01", "2023-12-31"),
        ]
        assert find_overlapping_periods(start_date, end_date) == expected

    def test_invalid_date_range(self):
        """Test case where the start date is after the end date."""
        start_date = datetime(2024, 12, 31)  # noqa: DTZ001
        end_date = datetime(2024, 1, 1)  # noqa: DTZ001
        with pytest.raises(ValueError, match="Start date must be before end date"):
            find_overlapping_periods(start_date, end_date)

    def test_same_start_and_end_date(self):
        """Test case where the start and end dates are the same."""
        start_date = datetime(2024, 6, 1)  # noqa: DTZ001
        end_date = datetime(2024, 6, 1)  # noqa: DTZ001
        expected = [("2024-06-01", "2024-06-01")]
        assert find_overlapping_periods(start_date, end_date) == expected

    def test_single_day_range(self):
        """Test case where the date range is a single day."""
        start_date = datetime(2025, 3, 5)  # noqa: DTZ001
        end_date = datetime(2025, 3, 5)  # noqa: DTZ001
        expected = [("2025-03-05", "2025-03-05")]
        assert find_overlapping_periods(start_date, end_date) == expected

    def test_large_date_range(self):
        """Test case with a very large date range spanning many years."""
        start_date = datetime(2000, 1, 1)  # noqa: DTZ001
        end_date = datetime(2025, 12, 31)  # noqa: DTZ001
        expected = [(f"{year}-01-01", f"{year}-12-31") for year in range(2000, 2026)]
        assert find_overlapping_periods(start_date, end_date) == expected
