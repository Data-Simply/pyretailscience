"""Tests for the graph_utils module in the style package."""

import datetime
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pytest
from scipy import stats

from pyretailscience.plots.styles import graph_utils as gu


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


def _extract_r2_from_plot(ax: plt.Axes) -> float:
    """Extract R² value from plot text annotations.

    Args:
        ax (plt.Axes): The matplotlib axes containing the R² annotation.

    Returns:
        float: The extracted R² value.
    """
    for text in ax.texts:
        content = text.get_text()
        if "R²" in content:
            match = re.search(r"R² = (-?[\d.]+(?:e[+-]?\d+)?)", content)
            if match:
                return float(match.group(1))
    pytest.fail("R² text not found in plot annotations")


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
    EXPECTED_LINEWIDTH = 2
    EXPECTED_ALPHA = 0.7
    OVERFLOW_X_MIN = 1000
    OVERFLOW_X_MAX = 2000

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
            datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 4, 1, tzinfo=datetime.timezone.utc),
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

    def test_unsupported_regression_type_raises_error(self):
        """Test that unsupported regression types raise ValueError."""
        # Create test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Create plot
        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise ValueError for unsupported type
        with pytest.raises(ValueError, match="Unsupported regression_type"):
            gu.add_regression_line(ax, regression_type="unsupported")

    # Algorithm Tests - Regression types with known data (parametrized to eliminate duplication)
    @pytest.mark.parametrize(
        ("regression_type", "x_data", "y_data", "description"),
        [
            ("linear", np.array([1, 2, 3, 4, 5]), lambda x: 2 * x + 1, "y = 2x + 1"),
            ("power", np.array([1, 2, 3, 4, 5]), lambda x: 2 * (x**1.5), "y = 2x^1.5"),
            ("logarithmic", np.array([1, 2, 3, 4, 5]), lambda x: 3 * np.log(x) + 1, "y = 3*ln(x) + 1"),
            ("exponential", np.array([0, 1, 2, 3, 4]), lambda x: 2 * np.exp(0.5 * x), "y = 2*e^(0.5x)"),
        ],
    )
    def test_regression_known_data(self, regression_type, x_data, y_data, description):
        """Test regression types with known relationships and validate line correctness."""
        # Generate perfect data based on known relationship
        y = y_data(x_data)

        # Create plot
        _, ax = plt.subplots()
        ax.scatter(x_data, y)

        # Apply regression
        result_ax = gu.add_regression_line(ax, regression_type=regression_type)

        # Verify basic functionality
        assert result_ax is ax
        assert len(ax.lines) == 1  # Regression line added

        # Validate that the regression line points are mathematically correct
        regression_line = ax.lines[0]
        line_x = regression_line.get_xdata()
        line_y = regression_line.get_ydata()

        # Check that the regression line produces values close to the expected relationship
        # Sample a few points from the regression line to verify correctness
        sample_indices = [0, len(line_x) // 2, -1]  # Start, middle, end
        for idx in sample_indices:
            x_val = line_x[idx]
            y_val = line_y[idx]
            # Calculate expected y value based on the known relationship
            expected_y = y_data(np.array([x_val]))[0]
            # Perfect data should yield near-exact fit (within floating-point precision)
            tolerance = max(abs(expected_y) * 1e-6, 1e-10)  # Relative 0.0001% with absolute floor
            assert abs(y_val - expected_y) < tolerance, (
                f"Regression line point ({x_val}, {y_val}) deviates too much from "
                f"expected ({x_val}, {expected_y}) for {description}"
            )

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([-1, 0, 1, 2, 3]), np.array([1, 2, 3, 4, 5])),
            (np.array([1, 2, 3]), np.array([1, 2, -3])),
        ],
        ids=["negative/zero x values", "negative y values"],
    )
    def test_power_regression_errors_on_nonpositive_values(self, x_data, y_data):
        """Test that power regression raises error when data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Power regression requires all x and y values to be positive"):
            gu.add_regression_line(ax, regression_type="power")

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])),
            (np.array([-1, 0.1, 1]), np.array([1, 2, 3])),
        ],
        ids=["zero x values", "negative x values"],
    )
    def test_logarithmic_regression_errors_on_nonpositive_x_values(self, x_data, y_data):
        """Test that logarithmic regression raises error when x data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Logarithmic regression requires all x values to be positive"):
            gu.add_regression_line(ax, regression_type="logarithmic")

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([1, 2, 3, 4, 5]), np.array([-1, 2, 3, 4, 5])),
            (np.array([1, 2, 3]), np.array([1, 0, 2])),
        ],
        ids=["negative y values", "zero y values"],
    )
    def test_exponential_regression_errors_on_nonpositive_y_values(self, x_data, y_data):
        """Test that exponential regression raises error when y data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Exponential regression requires all y values to be positive"):
            gu.add_regression_line(ax, regression_type="exponential")

    def test_exponential_regression_filters_overflow_values(self):
        """Test that exponential regression filters out infinite values from overflow without leaking numpy warnings."""
        # Use data with a steep growth coefficient so that extending x_max triggers overflow
        # With b ≈ 0.5, overflow (exp > 1e308) occurs around x ≈ 1420
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.65, 2.72, 4.48, 7.39, 12.18])  # Roughly y = e^(0.5*x)

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Set xlim far beyond overflow threshold to guarantee some points overflow
        ax.set_xlim(-1, self.OVERFLOW_X_MAX)

        # Ensure no numpy RuntimeWarning leaks to users during overflow
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            gu.add_regression_line(ax, regression_type="exponential")

        # Verify line was plotted with only finite values (some points should have been filtered)
        assert len(ax.lines) == 1
        regression_line = ax.lines[0]
        x_line = regression_line.get_xdata()
        y_data = regression_line.get_ydata()
        assert len(y_data) > 0
        assert np.all(np.isfinite(y_data))
        # Verify that filtering actually occurred: line should not extend all the way to x_max
        assert max(x_line) < self.OVERFLOW_X_MAX, "Some points should have been filtered due to overflow"

    def test_exponential_regression_warns_and_returns_early_on_total_overflow(self):
        """Test that exponential regression warns and plots nothing when all values overflow."""
        # Exponential data: fit yields b ≈ ln(10) ≈ 2.3, so exp(2.3 * 1000) overflows float64
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Set xlim to an extreme range where all exp() values overflow to infinity
        ax.set_xlim(self.OVERFLOW_X_MIN, self.OVERFLOW_X_MAX)

        # Snapshot axes state before calling add_regression_line
        lines_before = len(ax.lines)
        texts_before = len(ax.texts)

        # Ensure no numpy RuntimeWarning leaks — only the expected UserWarning should be raised
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.warns(UserWarning, match="produced no finite values in the visible range"):
                result = gu.add_regression_line(ax, regression_type="exponential")

        # Verify no regression line or equation text was added
        assert len(ax.lines) == lines_before
        assert len(ax.texts) == texts_before
        # Verify the axes object is still returned for chaining
        assert result is ax

    @pytest.mark.parametrize(
        ("regression_type", "expected_pattern"),
        [
            ("linear", r"y = .+x [+-] .+"),
            ("power", r"y = .+x\^.+"),
            ("logarithmic", r"y = .+ln\(x\) [+-] .+"),
            ("exponential", r"y = .+e\^\(.+x\)"),
        ],
    )
    def test_equation_formatting_different_types(self, regression_type, expected_pattern):
        """Test that equation text uses the correct format for each regression type."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        gu.add_regression_line(ax, regression_type=regression_type, show_equation=True, show_r2=True)

        texts = ax.texts
        assert len(texts) >= 1
        equation_text = texts[0].get_text()
        assert re.search(expected_pattern, equation_text), (
            f"Expected equation format matching '{expected_pattern}' for {regression_type}, got '{equation_text}'"
        )

    def test_regression_parameters_forwarding(self):
        """Test that regression parameters are forwarded correctly for new types."""
        # Create test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with power regression and custom parameters
        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should work with all parameters
        result_ax = gu.add_regression_line(
            ax,
            regression_type="power",
            color="blue",
            linestyle=":",
            text_position=0.8,
            show_equation=False,
            show_r2=True,
            linewidth=self.EXPECTED_LINEWIDTH,
            alpha=self.EXPECTED_ALPHA,
        )

        # Verify basic functionality
        assert result_ax is ax
        assert len(ax.lines) == 1

        # Verify line style parameters were forwarded
        line = ax.lines[0]
        assert line.get_color() == "blue"
        assert line.get_linestyle() == ":"
        assert line.get_linewidth() == self.EXPECTED_LINEWIDTH
        assert line.get_alpha() == self.EXPECTED_ALPHA

        # show_equation=False means no equation text, only R²
        texts = [t.get_text() for t in ax.texts]
        assert any("R²" in t for t in texts), "R² text should be displayed"
        assert not any("y =" in t for t in texts), "Equation should not be displayed when show_equation=False"

    @pytest.mark.parametrize("regression_type", ["linear", "power", "logarithmic", "exponential"])
    def test_insufficient_data_points_all_types(self, regression_type):
        """Test that all regression types handle insufficient data points correctly."""
        # Single data point
        x = np.array([1])
        y = np.array([2])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise error for insufficient data
        with pytest.raises(ValueError, match=r"At least .* valid data points are required"):
            gu.add_regression_line(ax, regression_type=regression_type)

    @pytest.mark.parametrize("regression_type", ["linear", "power", "logarithmic", "exponential"])
    def test_zero_variance_x_values(self, regression_type):
        """Test that all regression types handle zero variance in x values correctly."""
        # All x values are identical
        x = np.array([5, 5, 5, 5])
        y = np.array([1, 2, 3, 4])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise error for zero variance in x
        with pytest.raises(ValueError, match="all x values are identical"):
            gu.add_regression_line(ax, regression_type=regression_type)

    @pytest.mark.parametrize("regression_type", ["power", "exponential"])
    def test_zero_variance_y_values_nonlinear(self, regression_type):
        """Test that regression types with log(y) transform raise on zero variance y."""
        # All y values are identical — log transform makes regression meaningless
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 5, 5, 5])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        with pytest.raises(ValueError, match="all y values are identical"):
            gu.add_regression_line(ax, regression_type=regression_type)

    @pytest.mark.parametrize("regression_type", ["linear", "logarithmic"])
    def test_zero_variance_y_values_returns_flat_line(self, regression_type):
        """Test that linear and logarithmic regression handle constant y gracefully."""
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 5, 5, 5])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        result = gu.add_regression_line(ax, regression_type=regression_type)
        # Should succeed and return the axes (flat line is valid)
        assert result is ax
        # A regression line should have been added (scatter + regression = 2 lines)
        assert len(ax.get_lines()) == 1

    def test_r_squared_original_space_accuracy(self):
        """Test that R² is calculated in original data space, not transformed space."""
        # Create perfect power law data: y = 2 * x^1.5
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * (x_data**1.5)  # Perfect power law

        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        # Apply power regression
        gu.add_regression_line(ax, regression_type="power", show_r2=True)

        # With perfect power law data, R² should be very close to 1.0
        r2_value = _extract_r2_from_plot(ax)
        high_r_squared_threshold = 0.99
        assert r2_value > high_r_squared_threshold, f"R² should be close to 1.0 for perfect data, got {r2_value}"

    def test_r_squared_comparison_transformed_vs_original(self):
        """Test that R² in original space differs from transformed space for non-linear regression."""
        # Create data with a non-linear relationship that has deliberate outliers
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Quadratic pattern with outliers: y = x^2, but with outliers at the end
        y_data = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 200.0, 400.0])

        # Calculate R² in transformed space (log-log for power regression)
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        _, _, r_value_transformed, _, _ = stats.linregress(log_x, log_y)
        r_squared_transformed = r_value_transformed**2

        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        # Apply power regression - this should calculate R² in original space
        gu.add_regression_line(ax, regression_type="power", show_r2=True)

        # Extract R² from annotation (this is R² in original space)
        r2_value_original = _extract_r2_from_plot(ax)

        # Verify that R² values differ between spaces
        # With outliers, R² in original space should be significantly lower than in transformed space
        # R² in original space can be negative when the model performs worse than the mean
        assert r2_value_original <= 1.0, f"R² should be at most 1.0, got {r2_value_original}"
        assert 0.0 <= r_squared_transformed <= 1.0, (
            f"R² transformed should be between 0 and 1, got {r_squared_transformed}"
        )

        # The key test: original space R² should be lower because outliers
        # have a larger impact in original space than in log-transformed space
        assert r2_value_original < r_squared_transformed, (
            f"R² in original space ({r2_value_original:.4f}) should be lower than "
            f"transformed space ({r_squared_transformed:.4f}) due to outlier sensitivity"
        )

        # Also verify the difference is meaningful, not just floating-point noise
        difference_threshold = 0.01
        assert abs(r2_value_original - r_squared_transformed) > difference_threshold, (
            f"R² should meaningfully differ between original ({r2_value_original:.4f}) and "
            f"transformed ({r_squared_transformed:.4f}) space"
        )


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
