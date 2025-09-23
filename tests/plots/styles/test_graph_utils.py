"""Tests for the graph_utils module in the style package."""

import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pytest

from pyretailscience.plots.styles import graph_utils as gu


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


def test_human_format_basic():
    """Test basic human_format functionality."""
    assert gu.human_format(500) == "500"  # No suffix
    assert gu.human_format(1500) == "2K"  # Rounds to nearest thousand
    assert gu.human_format(1500000) == "2M"  # Rounds to nearest million


def test_human_format_with_decimals():
    """Test human_format with decimals."""
    assert gu.human_format(1500, decimals=2) == "1.5K"
    assert gu.human_format(1500000, decimals=1) == "1.5M"
    assert gu.human_format(1500000, decimals=3) == "1.5M"
    assert gu.human_format(1234567, decimals=3) == "1.235M"
    assert gu.human_format(1234567, decimals=4) == "1.2346M"


def test_human_format_with_prefix():
    """Test human_format with a prefix."""
    assert gu.human_format(1500, prefix="$") == "$2K"
    assert gu.human_format(1500000, prefix="€") == "€2M"
    assert gu.human_format(500, prefix="¥") == "¥500"


def test_human_format_magnitude_promotion():
    """Test human_format with magnitude promotion."""
    assert gu.human_format(1000000) == "1M"
    assert gu.human_format(1000000000) == "1B"
    assert gu.human_format(1000, decimals=2) == "1K"  # Does not promote when unnecessary


def test_human_format_edge_zero():
    """Test human_format with edge cases involving zero."""
    assert gu.human_format(0) == "0"


def test_human_format_negative_numbers():
    """Test human_format with negative numbers."""
    assert gu.human_format(-1500) == "-2K"
    assert gu.human_format(-1500000, decimals=1) == "-1.5M"
    assert gu.human_format(-1234567, decimals=3) == "-1.235M"
    assert gu.human_format(-1000000000, decimals=2) == "-1B"


def test_human_format_very_small_numbers():
    """Test human_format with very small numbers."""
    assert gu.human_format(0.001) == "0"  # No suffix, rounds to 0
    assert gu.human_format(999.999, decimals=2) == "1K"  # Just below 1000 but rounds up


def test_human_format_large_numbers():
    """Test human_format with very large numbers."""
    assert gu.human_format(10**15) == "1P"  # P for petabyte scale numbers
    assert gu.human_format(10**17) == "100P"  # Even larger, stays in petabyte scale


def test_human_format_no_suffix_needed():
    """Test human_format with numbers that don't need a suffix."""
    assert gu.human_format(999) == "999"
    assert gu.human_format(500) == "500"


def test_human_format_exactly_1000():
    """Test human_format with numbers that are exactly multiples of 1000."""
    assert gu.human_format(1000) == "1K"
    assert gu.human_format(1000000) == "1M"
    assert gu.human_format(1000000000) == "1B"


def test_human_format_multiple_promotions():
    """Test human_format with multiple magnitude promotions."""
    assert gu.human_format(1000000000) == "1B"  # 1,000,000,000 -> 1B
    assert gu.human_format(1000000000000) == "1T"  # 1,000,000,000,000 -> 1T


def test_human_format_decimal_rounding():
    """Test human_format with decimal rounding."""
    assert gu.human_format(1234567, decimals=4) == "1.2346M"  # Rounding to four decimals
    assert gu.human_format(1234567, decimals=2) == "1.23M"  # Rounding to two decimals
    assert gu.human_format(1234567, decimals=0) == "1M"  # No decimals


def test_human_format_suffix_upper_bound():
    """Test human_format with the largest suffix provided."""
    assert gu.human_format(10**15) == "1P"  # Largest suffix provided is "P"
    assert gu.human_format(10**16) == "10P"  # Stay in P range


def test_human_format_negative_magnitude_promotion():
    """Test human_format with negative numbers that promote magnitude."""
    assert gu.human_format(-1000000) == "-1M"
    assert gu.human_format(-1000000000) == "-1B"
    assert gu.human_format(-1000) == "-1K"


def test_human_format_decimal_edge_cases():
    """Test human_format with edge cases involving decimals."""
    assert gu.human_format(999.999, decimals=0) == "1K"  # Rounds up to 1000
    assert gu.human_format(999999.999, decimals=0) == "1M"  # Rounds to next magnitude
    assert gu.human_format(1000.0, decimals=0) == "1K"  # Exactly at boundary


def test_truncate_to_x_digits_basic():
    """Test basic truncate_to_x_digits functionality."""
    assert gu.truncate_to_x_digits("1.5K", 2) == "1.5K"
    assert gu.truncate_to_x_digits("1.25M", 3) == "1.25M"
    assert gu.truncate_to_x_digits("1M", 1) == "1M"
    assert gu.truncate_to_x_digits("10.25M", 3) == "10.2M"
    assert gu.truncate_to_x_digits("10.25M", 4) == "10.25M"
    assert gu.truncate_to_x_digits("10.99M", 3) == "10.9M"
    assert gu.truncate_to_x_digits("1.234K", 2) == "1.2K"
    assert gu.truncate_to_x_digits("5.678M", 3) == "5.67M"
    assert gu.truncate_to_x_digits("9.999B", 2) == "9.9B"


def test_truncate_to_x_digits_number_greater_than_digits():
    """Test truncate_to_x_digits with number greater than digits."""
    assert gu.truncate_to_x_digits("500", 2) == "500"
    assert gu.truncate_to_x_digits("12345", 3) == "12345"


def test_truncate_to_x_digits_edge_zero():
    """Test truncate_to_x_digits with edge cases involving zero."""
    assert gu.truncate_to_x_digits("0", 2) == "0"
    assert gu.truncate_to_x_digits("0K", 2) == "0K"


def test_truncate_to_x_digits_negative_numbers():
    """Test truncate_to_x_digits with negative numbers."""
    assert gu.truncate_to_x_digits("-1.5K", 2) == "-1.5K"
    assert gu.truncate_to_x_digits("-1.234M", 3) == "-1.23M"


def test_truncate_to_x_digits_very_small_numbers():
    """Test truncate_to_x_digits with very small numbers."""
    assert gu.truncate_to_x_digits("0.001", 2) == "0"
    assert gu.truncate_to_x_digits("0.000009", 7) == "0.000009"


def test_truncate_to_x_digits_large_numbers():
    """Test truncate_to_x_digits with very large numbers."""
    assert gu.truncate_to_x_digits("1.234B", 4) == "1.234B"  # Truncate large numbers
    assert gu.truncate_to_x_digits("1.234P", 2) == "1.2P"  # Truncate large numbers with suffix


def test_truncate_to_x_digits_no_truncation_needed():
    """Test truncate_to_x_digits with no truncation needed."""
    assert gu.truncate_to_x_digits("123", 3) == "123"
    assert gu.truncate_to_x_digits("12.345", 5) == "12.345"


def test_truncate_to_x_digits_exact_digits():
    """Test truncate_to_x_digits with exact number of digits."""
    assert gu.truncate_to_x_digits("999", 3) == "999"
    assert gu.truncate_to_x_digits("1.234M", 4) == "1.234M"


def test_truncate_to_x_digits_trailing_zeros():
    """Test truncate_to_x_digits with trailing zeros."""
    assert gu.truncate_to_x_digits("1.500", 3) == "1.5"
    assert gu.truncate_to_x_digits("1.230K", 4) == "1.23K"  # Removes trailing zero
    assert gu.truncate_to_x_digits("10.000", 2) == "10"


def test_truncate_to_x_digits_decimal_edge_cases():
    """Test truncate_to_x_digits with edge cases involving decimals."""
    assert gu.truncate_to_x_digits("0.9999", 3) == "0.99"
    assert gu.truncate_to_x_digits("999.999K", 4) == "999.9K"
    assert gu.truncate_to_x_digits("100.0001M", 4) == "100M"


def test_set_axis_percent():
    """Test set_axis_percent function formats axis correctly."""
    # Create a test plot
    _, ax = plt.subplots()
    ax.plot([0, 0.25, 0.5, 0.75, 1.0], [0, 0.3, 0.5, 0.7, 1.0])

    # Apply our function to the y-axis
    gu.set_axis_percent(ax.yaxis)

    # Check that the formatter is properly applied
    assert isinstance(ax.yaxis.get_major_formatter(), mtick.PercentFormatter)

    # Check default parameters
    formatter = ax.yaxis.get_major_formatter()
    assert formatter.xmax == 1
    assert formatter._symbol == "%"

    # Test with custom parameters
    _, ax = plt.subplots()
    ax.plot([0, 25, 50, 75, 100], [0, 30, 50, 70, 100])

    # Define test values
    test_xmax = 100
    test_decimals = 2
    test_symbol = "pct"

    # Apply with custom parameters
    gu.set_axis_percent(ax.xaxis, xmax=test_xmax, decimals=test_decimals, symbol=test_symbol)

    # Check that the formatter is properly applied with custom params
    formatter = ax.xaxis.get_major_formatter()
    assert isinstance(formatter, mtick.PercentFormatter)
    assert formatter.xmax == test_xmax
    assert formatter.decimals == test_decimals
    assert formatter._symbol == test_symbol


class TestRegressionLine:
    """Test class for the add_regression_line function."""

    # Constants to avoid magic numbers
    ORIGINAL_LINE_COUNT = 1
    EXPECTED_LINE_COUNT_AFTER_REGRESSION = 2
    STACKED_PATCH_COUNT = 8  # 4 bars + 4 stacked bars
    GROUPED_PATCH_COUNT = 8  # 4 bars + 4 grouped bars
    REGRESSION_LINE_POINTS = 2  # Regression line has 2 endpoints

    def test_line_plot_with_numeric_data(self):
        """Test regression line with a standard line plot and numeric data."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 5, 7, 11])  # Not a perfect line to test regression
        ax.plot(x, y)

        gu.add_regression_line(ax, color="blue", show_equation=True, show_r2=True)

        # Check that a line was added (should now have 2 lines)
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_line_plot_with_datetime_data(self):
        """Test regression line with datetime x-axis data."""
        _, ax = plt.subplots()
        dates = [
            datetime.datetime(2023, 1, 1, tzinfo=datetime.UTC),
            datetime.datetime(2023, 2, 1, tzinfo=datetime.UTC),
            datetime.datetime(2023, 3, 1, tzinfo=datetime.UTC),
            datetime.datetime(2023, 4, 1, tzinfo=datetime.UTC),
        ]
        values = [10, 15, 14, 25]
        ax.plot(dates, values)

        gu.add_regression_line(ax, show_equation=True, show_r2=False)

        # Check that a line was added
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_scatter_plot(self):
        """Test regression line with a scatter plot."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3.5, 4.5, 7.5, 10])
        ax.scatter(x, y)

        gu.add_regression_line(ax, color="green", linestyle="-.")

        # Check that a line was added to the scatter plot
        assert len(ax.get_lines()) == self.ORIGINAL_LINE_COUNT

    def test_large_numbers(self):
        """Test regression line with very large numbers (billions)."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.5, 3.2, 4.7, 7.1, 8.9]) * 1e9  # Values in billions
        ax.plot(x, y)

        gu.add_regression_line(ax, color="purple", show_equation=True, show_r2=True)

        # Check that a line was added
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_bar_plot(self):
        """Test regression line with a vertical bar chart."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 3, 5, 6])
        ax.bar(x, y)

        gu.add_regression_line(ax, color="orange", show_equation=True, show_r2=True)

        # Check that a regression line was added (bar plots start with 0 lines)
        assert len(ax.get_lines()) == 1
        # Check that we still have the bar patches
        assert len(ax.patches) == len(x)

    def test_barh_plot(self):
        """Test regression line with a horizontal bar chart."""
        _, ax = plt.subplots()
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([2, 4, 3, 5, 6])
        ax.barh(y, x)

        gu.add_regression_line(ax, color="green", show_equation=True, show_r2=True)

        # Check that a regression line was added
        assert len(ax.get_lines()) == 1
        # Check that we still have the bar patches
        assert len(ax.patches) == len(x)

    def test_single_data_point(self):
        """Test that regression line raises ValueError with a single data point."""
        _, ax = plt.subplots()
        ax.plot([10], [20])

        # Use pytest.raises to check for the expected exception
        with pytest.raises(ValueError) as excinfo:
            gu.add_regression_line(ax)

        # Check for appropriate error message
        assert "regression" in str(excinfo.value).lower()

    def test_bar_plot_negative_values(self):
        """Test regression line correctly handles negative bar values."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([-2, 4, -1, 3, -5])  # Mix of positive and negative values
        ax.bar(x, y)

        gu.add_regression_line(ax, color="red")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Get the regression line data
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # Verify the line spans a reasonable range (uses axis limits, not exact data range)
        assert line_x[0] < min(x)  # Line starts before first bar
        assert line_x[1] > max(x)  # Line ends after last bar

        # Verify line handles negative values (should not be all zeros)
        assert not all(val == 0 for val in line_y)

    def test_barh_plot_negative_values(self):
        """Test regression line correctly handles negative horizontal bar values."""
        _, ax = plt.subplots()
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([-2, 4, -1, 3, -5])  # Mix of positive and negative values
        ax.barh(y, x)

        gu.add_regression_line(ax, color="purple")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Get the regression line data
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # For horizontal bars, verify the line spans the value range reasonably
        # Line should encompass the data range (may extend beyond due to axis limits)
        assert min(line_x) <= max(x)  # Line should reach at least the max value
        assert max(line_x) >= min(x)  # Line should reach at least the min value

        # Verify line handles negative values (should not be all zeros)
        assert not all(val == 0 for val in line_y)

    def test_bar_plot_stacked(self):
        """Test regression line with stacked bar chart uses correct data ordering."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4])
        y1 = np.array([2, 3, 4, 1])
        y2 = np.array([1, 2, 1, 3])

        # Create stacked bars
        ax.bar(x, y1, label="Series 1")
        ax.bar(x, y2, bottom=y1, label="Series 2")

        gu.add_regression_line(ax, color="blue")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Verify we have patches for both series (8 total: 4 + 4)
        assert len(ax.patches) == self.STACKED_PATCH_COUNT

        # Get the regression line to ensure it was calculated
        line = ax.get_lines()[0]
        line_x = line.get_xdata()

        # Verify line spans the x range of the bars
        assert min(line_x) <= min(x)
        assert max(line_x) >= max(x)

    def test_bar_plot_grouped(self):
        """Test regression line with grouped bar chart handles data ordering correctly."""
        _, ax = plt.subplots()

        # Create grouped bars with different x positions
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array([1.3, 2.3, 3.3, 4.3])  # Offset for grouping
        y1 = np.array([2, 3, 4, 1])
        y2 = np.array([3, 1, 2, 4])

        width = 0.3
        ax.bar(x1, y1, width, label="Group 1")
        ax.bar(x2, y2, width, label="Group 2")

        gu.add_regression_line(ax, color="orange")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Verify we have patches for both groups (8 total: 4 + 4)
        assert len(ax.patches) == self.GROUPED_PATCH_COUNT

        # Get the regression line
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # Verify line data exists and spans a reasonable range
        assert len(line_x) == self.REGRESSION_LINE_POINTS  # Regression line should have 2 points
        assert len(line_y) == self.REGRESSION_LINE_POINTS

        # Verify line spans across the grouped bars
        all_x_positions = np.concatenate([x1, x2])
        assert min(line_x) <= min(all_x_positions)
        assert max(line_x) >= max(all_x_positions)

    def test_bar_plot_container_no_orientation_attr(self):
        """Test regression line with container lacking orientation attribute."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 3])
        ax.bar(x, y)

        # Add a mock container without orientation to test the hasattr branch
        class MockContainer:
            pass

        mock_container = MockContainer()
        ax.containers.insert(0, mock_container)  # Insert at beginning

        gu.add_regression_line(ax, color="red")
        # Should still work by falling back to default (vertical)
        assert len(ax.get_lines()) == 1

    def test_bar_plot_container_none_orientation(self):
        """Test regression line with container having None orientation."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 3])
        ax.bar(x, y)

        ax.containers[0].orientation = None

        gu.add_regression_line(ax, color="blue")
        # Should still work by falling back to default (vertical)
        assert len(ax.get_lines()) == 1


class TestVisualRegression:
    """Visual regression tests to ensure refactored code produces identical output."""

    WHITE_RGBA = (1.0, 1.0, 1.0, 1.0)

    def test_visual_regression_standard_graph_styles_basic(self):
        """Ensure standard_graph_styles produces consistent visual output."""
        fig, ax = plt.subplots(figsize=(8, 6))
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.1 * rng.standard_normal(50)
        ax.plot(x, y, label="Sin Wave")

        gu.standard_graph_styles(
            ax,
            title="Test Graph Title",
            x_label="X Axis Label",
            y_label="Y Axis Label",
            legend_title="Legend Title",
        )

        assert ax.get_title() == "Test Graph Title"
        assert ax.get_xlabel() == "X Axis Label"
        assert ax.get_ylabel() == "Y Axis Label"
        assert ax.get_facecolor() == self.WHITE_RGBA
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

        plt.close(fig)

    @pytest.mark.parametrize(
        ("plot_type", "data_generator"),
        [
            (
                "line",
                lambda: (
                    np.linspace(0, 10, 20),
                    np.sin(np.linspace(0, 10, 20)) + 0.1 * np.random.default_rng(42).standard_normal(20),
                ),
            ),
            (
                "scatter",
                lambda: (
                    np.linspace(0, 10, 20),
                    np.sin(np.linspace(0, 10, 20)) + 0.1 * np.random.default_rng(42).standard_normal(20),
                ),
            ),
            ("bar", lambda: (["A", "B", "C", "D", "E"], [23, 45, 56, 78, 32])),
        ],
    )
    def test_visual_regression_all_plot_types(self, plot_type, data_generator):
        """Test visual consistency across different plot types."""
        fig, ax = plt.subplots(figsize=(8, 6))

        x, y = data_generator()

        plot_methods = {
            "line": lambda ax, x, y: ax.plot(x, y),
            "scatter": lambda ax, x, y: ax.scatter(x, y),
            "bar": lambda ax, x, y: ax.bar(x, y),
        }
        plot_methods[plot_type](ax, x, y)

        gu.standard_graph_styles(
            ax,
            title=f"Test {plot_type.title()} Plot",
            x_label="X Values",
            y_label="Y Values",
        )

        assert ax.get_facecolor() == self.WHITE_RGBA
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

        plt.close(fig)

    def test_visual_regression_source_text(self):
        """Test source text visual consistency."""
        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.linspace(0, 10, 50)
        y = np.exp(-x / 5) * np.cos(x)
        ax.plot(x, y)

        source_text = gu.add_source_text(ax, "Source: Test Data 2024")

        assert source_text.get_text() == "Source: Test Data 2024"
        assert source_text.get_color() == "dimgray"

        plt.close(fig)
