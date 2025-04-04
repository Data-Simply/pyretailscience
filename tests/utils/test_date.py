"""Tests for the time_utils module."""

# flake8: noqa: DTZ001
import datetime

import ibis
import pandas as pd
import pytest

from pyretailscience.utils.date import filter_and_label_by_periods, find_overlapping_periods


class TestFilterAndLabelByPeriods:
    """Test cases for filtering and labeling periods function."""

    @pytest.fixture
    def sample_transactions_table(self):
        """Fixture to provide a sample transactions table for testing."""
        data = {
            "transaction_id": [1, 2, 3, 4, 5, 6],
            "transaction_date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-02-10",
                    "2023-04-05",
                    "2023-05-20",
                    "2023-07-10",
                    "2023-08-25",
                ],
            ),
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0, 175.0],
        }
        df = pd.DataFrame(data)
        return ibis.memtable(df)

    def test_filter_and_label_by_periods_with_datetime_objects(self, sample_transactions_table):
        """Test filtering and labeling periods using datetime objects."""
        # Define expected transaction counts
        q1_count = 2
        q2_count = 2
        total_count = q1_count + q2_count

        period_ranges = {
            "Q1": (
                datetime.datetime(2023, 1, 1, tzinfo=datetime.UTC),
                datetime.datetime(2023, 3, 31, tzinfo=datetime.UTC),
            ),
            "Q2": (
                datetime.datetime(2023, 4, 1, tzinfo=datetime.UTC),
                datetime.datetime(2023, 6, 30, tzinfo=datetime.UTC),
            ),
        }
        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        # Verify the correct number of transactions
        assert len(result_df) == total_count, f"Should return {total_count} transactions from Q1 and Q2"

        # Verify the period labels are correct
        q1_transactions = result_df[result_df["period_name"] == "Q1"]
        q2_transactions = result_df[result_df["period_name"] == "Q2"]

        assert len(q1_transactions) == q1_count, f"Should have {q1_count} transactions in Q1"
        assert len(q2_transactions) == q2_count, f"Should have {q2_count} transactions in Q2"

        # Verify the transaction IDs are correctly assigned to periods
        assert sorted(q1_transactions["transaction_id"].tolist()) == [1, 2], "Q1 should have transaction IDs 1 and 2"
        assert sorted(q2_transactions["transaction_id"].tolist()) == [3, 4], "Q2 should have transaction IDs 3 and 4"

    def test_filter_and_label_by_periods_with_string_dates(self, sample_transactions_table):
        """Test filtering and labeling periods using string dates."""
        # Define expected transaction counts
        q1_count = 2
        q2_count = 2
        q3_count = 2
        total_count = q1_count + q2_count + q3_count

        period_ranges = {
            "Q1": ("2023-01-01", "2023-03-31"),
            "Q2": ("2023-04-01", "2023-06-30"),
            "Q3": ("2023-07-01", "2023-09-30"),
        }

        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        # Verify the correct number of transactions
        assert len(result_df) == total_count, f"Should return all {total_count} transactions across Q1, Q2, and Q3"

        # Verify the period labels are correct for each quarter
        q1_transactions = result_df[result_df["period_name"] == "Q1"]
        q2_transactions = result_df[result_df["period_name"] == "Q2"]
        q3_transactions = result_df[result_df["period_name"] == "Q3"]

        assert len(q1_transactions) == q1_count, f"Should have {q1_count} transactions in Q1"
        assert len(q2_transactions) == q2_count, f"Should have {q2_count} transactions in Q2"
        assert len(q3_transactions) == q3_count, f"Should have {q3_count} transactions in Q3"

        # Verify specific transaction IDs are in the correct periods
        assert sorted(q1_transactions["transaction_id"].tolist()) == [1, 2], "Q1 should have transaction IDs 1 and 2"
        assert sorted(q2_transactions["transaction_id"].tolist()) == [3, 4], "Q2 should have transaction IDs 3 and 4"
        assert sorted(q3_transactions["transaction_id"].tolist()) == [5, 6], "Q3 should have transaction IDs 5 and 6"

    def test_filter_and_label_by_periods_invalid_period_format(self, sample_transactions_table):
        """Test that invalid period formats raise ValueError."""
        # Single date instead of tuple
        period_ranges = {"Invalid": "2023-01-01"}

        with pytest.raises(ValueError, match="Period 'Invalid' must have a \\(start_date, end_date\\) tuple"):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

        # Tuple with wrong length
        period_ranges = {"Invalid": ("2023-01-01", "2023-03-31", "extra-value")}

        with pytest.raises(ValueError, match="Period 'Invalid' must have a \\(start_date, end_date\\) tuple"):
            filter_and_label_by_periods(sample_transactions_table, period_ranges)

    def test_filter_and_label_by_periods_no_matches(self, sample_transactions_table):
        """Test behavior when no transactions match the given periods."""
        period_ranges = {
            "Future": ("2024-01-01", "2024-12-31"),
        }

        result = filter_and_label_by_periods(sample_transactions_table, period_ranges)
        result_df = result.execute()

        # Should return an empty dataframe since no transactions match the period
        assert len(result_df) == 0, "Should return 0 transactions for future period"


class TestFindOverlappingPeriods:
    """Test cases for the find_overlapping_periods function."""

    def test_valid_range_same_year(self):
        """Test case where the start and end dates are in the same year."""
        start_date = datetime.datetime(2024, 1, 1)
        end_date = datetime.datetime(2024, 12, 31)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_multiple_years(self):
        """Test case where the start and end dates span multiple years."""
        start_date = datetime.datetime(2022, 6, 15)
        end_date = datetime.datetime(2025, 3, 10)
        expected = [
            ("2022-06-15", "2023-03-10"),
            ("2023-06-15", "2024-03-10"),
            ("2024-06-15", "2025-03-10"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_full_years(self):
        """Test case with a date range spanning full years."""
        start_date = datetime.datetime(2020, 1, 1)
        end_date = datetime.datetime(2023, 12, 31)
        expected = [
            ("2020-01-01", "2021-12-31"),
            ("2021-01-01", "2022-12-31"),
            ("2022-01-01", "2023-12-31"),
        ]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_invalid_date_range(self):
        """Test case where the start date is after the end date."""
        start_date = datetime.datetime(2024, 12, 31)
        end_date = datetime.datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="Start date must be before end date"):
            find_overlapping_periods(start_date, end_date)

    def test_same_start_and_end_date(self):
        """Test case where the start and end dates are the same."""
        start_date = datetime.datetime(2024, 6, 1)
        end_date = datetime.datetime(2024, 6, 1)
        expected = []
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_large_date_range(self):
        """Test case with a very large date range spanning many years."""
        start_date = datetime.datetime(2000, 1, 1)
        end_date = datetime.datetime(2025, 12, 31)
        expected = [(f"{year}-01-01", f"{year + 1}-12-31") for year in range(2000, 2025)]
        result = find_overlapping_periods(start_date, end_date)
        assert result == expected

    def test_valid_range_multiple_years_datetime(self):
        """Test case where the start and end dates span multiple years with return_iso=False."""
        start_date = datetime.datetime(2021, 5, 10)
        end_date = datetime.datetime(2024, 8, 20)
        expected = [
            (datetime.datetime(2021, 5, 10), datetime.datetime(2022, 8, 20)),
            (datetime.datetime(2022, 5, 10), datetime.datetime(2023, 8, 20)),
            (datetime.datetime(2023, 5, 10), datetime.datetime(2024, 8, 20)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected

    def test_valid_range_multiple_years_string(self):
        """Test case where the start and end dates are provided as strings with return_iso=False."""
        start_date = "2021-05-10"
        end_date = "2024-08-20"
        expected = [
            (datetime.datetime(2021, 5, 10), datetime.datetime(2022, 8, 20)),
            (datetime.datetime(2022, 5, 10), datetime.datetime(2023, 8, 20)),
            (datetime.datetime(2023, 5, 10), datetime.datetime(2024, 8, 20)),
        ]
        result = find_overlapping_periods(start_date, end_date, return_str=False)
        assert result == expected
