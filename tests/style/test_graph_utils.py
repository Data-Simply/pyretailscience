"""Tests for the graph_utils module in the style package."""

from pyretailscience.style import graph_utils as gu


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
