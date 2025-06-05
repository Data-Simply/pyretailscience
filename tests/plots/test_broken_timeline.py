"""Tests for the broken timeline plot module."""

from unittest.mock import patch

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pyretailscience.options import get_option
from pyretailscience.plots import broken_timeline


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing with intentional gaps."""
    date_col = get_option("column.transaction_date")

    # Create data with gaps for different categories
    data = {
        date_col: pd.to_datetime(
            [
                "2025-04-01",
                "2025-04-02",
                "2025-04-03",  # Category A: continuous
                "2025-04-06",
                "2025-04-07",  # Category A: gap then continues
                "2025-04-01",
                "2025-04-02",  # Category B: starts same time
                "2025-04-05",
                "2025-04-06",  # Category B: gap then continues
                "2025-04-03",
                "2025-04-04",
                "2025-04-05",  # Category C: different pattern
            ],
        ),
        "category": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C"],
        "value": [100, 150, 200, 120, 180, 300, 250, 400, 350, 80, 90, 110],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_single_category():
    """A sample dataframe with a single category for testing."""
    date_col = get_option("column.transaction_date")

    data = {
        date_col: pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-05", "2025-04-06"]),
        "category": ["Store1", "Store1", "Store1", "Store1"],
        "value": [100, 150, 200, 120],
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe():
    """An empty dataframe for testing."""
    date_col = get_option("column.transaction_date")
    return pd.DataFrame(columns=[date_col, "category", "value"])


class TestBrokenTimelinePlot:
    """Test cases for the broken timeline plot function."""

    def test_basic_functionality_and_labels(self, sample_dataframe):
        """Test basic plot creation and custom labels."""
        title = "Data Availability Timeline"
        x_label = "Custom Date Label"
        y_label = "Custom Category Label"

        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert isinstance(ax, Axes)
        assert ax.get_title() == title
        assert ax.get_xlabel() == x_label
        assert ax.get_ylabel() == y_label

    def test_single_category(self, sample_dataframe_single_category):
        """Test with a single category."""
        ax = broken_timeline.plot(
            df=sample_dataframe_single_category,
            category_col="category",
            value_col="value",
        )

        assert isinstance(ax, Axes)
        assert len(ax.get_yticklabels()) == 1

    def test_threshold_filtering(self, sample_dataframe):
        """Test threshold value filtering removes low values."""
        # Plot with threshold that should filter out some data
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            threshold_value=150,
        )

        # Plot without threshold for comparison
        ax_no_threshold = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
        )

        # With threshold, there should be fewer or equal bars than without threshold
        threshold_bars = len(list(ax.patches))
        no_threshold_bars = len(list(ax_no_threshold.patches))
        assert threshold_bars <= no_threshold_bars

    def test_different_periods(self):
        """Test period aggregation works correctly for valid periods."""
        date_col = get_option("column.transaction_date")

        # Create data spanning multiple weeks for comprehensive testing
        data = {
            date_col: pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-08",
                    "2025-01-15",
                    "2025-01-22",
                    "2025-01-29",
                    "2025-02-05",
                    "2025-02-12",
                ],
            ),
            "category": ["A"] * 8,
            "value": [100] * 8,
        }
        df = pd.DataFrame(data)

        results = {}
        for period in ["D", "W"]:
            ax = broken_timeline.plot(df, "category", "value", period=period)
            results[period] = len(list(ax.patches))

        # Weekly aggregation should have fewer or equal bars than daily
        assert results["W"] <= results["D"]

    def test_lowercase_period_handling(self, sample_dataframe):
        """Test that lowercase period parameters are properly converted."""
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            period="d",
        )
        assert isinstance(ax, Axes)

    def test_with_source_text(self, sample_dataframe):
        """Test adding source text appears in plot."""
        source_text = "Source: Test Data"
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            source_text=source_text,
        )

        # Check that source text appears in the plot's text elements
        text_elements = [text.get_text() for text in ax.figure.findobj(plt.Text)]
        assert source_text in text_elements

    def test_custom_axes(self, sample_dataframe):
        """Test plotting on a custom axes object."""
        fig, custom_ax = plt.subplots()

        result_ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            ax=custom_ax,
        )

        assert result_ax is custom_ax

    def test_empty_dataframe_raises_error(self, empty_dataframe):
        """Test that empty dataframe raises ValueError."""
        with pytest.raises(ValueError, match="Cannot plot with empty DataFrame"):
            broken_timeline.plot(
                df=empty_dataframe,
                category_col="category",
                value_col="value",
            )

    def test_missing_column_raises_error(self, sample_dataframe):
        """Test that missing columns raise KeyError."""
        with pytest.raises(KeyError, match="Required column 'nonexistent' not found"):
            broken_timeline.plot(
                df=sample_dataframe,
                category_col="nonexistent",
                value_col="value",
            )

    def test_invalid_period_raises_error(self, sample_dataframe):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Invalid period 'X'. Must be one of"):
            broken_timeline.plot(
                df=sample_dataframe,
                category_col="category",
                value_col="value",
                period="X",
            )

    def test_kwargs_passed_to_broken_barh(self, sample_dataframe):
        """Test that additional kwargs are passed to broken_barh."""
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            alpha=0.5,
            edgecolors="black",
        )

        assert isinstance(ax, Axes)

    def test_y_axis_inverted(self, sample_dataframe):
        """Test that y-axis is inverted (categories from top to bottom)."""
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
        )

        # Check that y-axis is inverted
        y_lim = ax.get_ylim()
        assert y_lim[0] > y_lim[1]  # First value should be greater than second for inverted axis

    def test_duplicate_date_category_combinations(self):
        """Test handling of duplicate date-category combinations."""
        date_col = get_option("column.transaction_date")

        # Create data with duplicates
        data = {
            date_col: pd.to_datetime(["2025-04-01", "2025-04-01", "2025-04-02"]),
            "category": ["A", "A", "A"],  # Same category, same date for first two rows
            "value": [100, 200, 150],  # Different values
        }
        df_with_duplicates = pd.DataFrame(data)

        ax = broken_timeline.plot(
            df=df_with_duplicates,
            category_col="category",
            value_col="value",
        )

        assert isinstance(ax, Axes)

    def test_categories_sorted_on_y_axis(self, sample_dataframe):
        """Test that categories are sorted on the y-axis."""
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
        )

        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        assert y_labels == sorted(y_labels)  # Should be sorted

    def test_no_data_for_category(self):
        """Test handling when a category has no data after filtering."""
        date_col = get_option("column.transaction_date")

        data = {
            date_col: pd.to_datetime(["2025-04-01", "2025-04-02"]),
            "category": ["A", "B"],
            "value": [50, 200],  # A will be filtered out with threshold 100
        }
        df = pd.DataFrame(data)

        ax = broken_timeline.plot(
            df=df,
            category_col="category",
            value_col="value",
            threshold_value=100,
        )

        assert isinstance(ax, Axes)

    @pytest.mark.parametrize(
        ("period", "dates", "num_periods"),
        [
            ("D", ["2025-01-01", "2025-01-02", "2025-01-03"], 3),
            ("W", ["2025-01-01", "2025-01-08"], 2),
        ],
    )
    def test_bar_width_calculation_for_different_periods(self, period, dates, num_periods):
        """Test that bar widths are calculated correctly for different time periods."""
        date_col = get_option("column.transaction_date")
        expected_width = num_periods * broken_timeline.PERIOD_CONFIG[period]

        data = {
            date_col: pd.to_datetime(dates),
            "category": ["A"] * len(dates),
            "value": [100] * len(dates),
        }
        df = pd.DataFrame(data)

        with patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            broken_timeline.plot(df, "category", "value", period=period)
            segments = mock_broken_barh.call_args[0][0]
            actual_width = segments[0][1]
            assert actual_width == expected_width

    @pytest.mark.parametrize(
        ("period", "dates", "expected_segments"),
        [
            (
                "D",
                ["2025-01-01", "2025-01-02", "2025-01-06", "2025-01-07"],
                2,  # 4-day gap > 1-day threshold creates 2 segments
            ),
            (
                "W",
                ["2025-01-01", "2025-01-08", "2025-01-22", "2025-01-29"],
                2,  # 14-day gap > 7-day threshold creates 2 segments
            ),
        ],
    )
    def test_gap_detection_with_different_periods(self, period, dates, expected_segments):
        """Test that gaps are correctly detected based on period type."""
        date_col = get_option("column.transaction_date")

        data = {
            date_col: pd.to_datetime(dates),
            "category": ["A"] * len(dates),
            "value": [100] * len(dates),
        }
        df = pd.DataFrame(data)

        with patch("matplotlib.axes.Axes.broken_barh") as mock_broken_barh:
            broken_timeline.plot(df, "category", "value", period=period)
            segments = mock_broken_barh.call_args[0][0]
            assert len(segments) == expected_segments
