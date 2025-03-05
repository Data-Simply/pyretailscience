"""Tests for the GainLoss class in the gain_loss module."""

import pytest

from pyretailscience.analysis.gain_loss import GainLoss


@pytest.mark.parametrize(
    ("focus_p1", "comparison_p1", "focus_p2", "comparison_p2", "focus_diff", "comparison_diff", "expected"),
    [
        # Test when a new customer is added (focus_p1 and comparison_p1 are 0)
        (0, 0, 100, 0, 100, 0, (100, 0, 0, 0, 0, 0)),
        # Test when a customer is lost (focus_p2 and comparison_p2 are 0)
        (100, 0, 0, 0, -100, 0, (0, -100, 0, 0, 0, 0)),
        # Test when both focus and comparison increase, but focus increase is greater
        (100, 100, 150, 120, 50, 20, (0, 0, 50, 0, 0, 0)),
        # Test when both focus and comparison increase, but comparison increase is greater
        (100, 100, 120, 150, 20, 50, (0, 0, 20, 0, 0, 0)),
        # Test when focus increases and comparison decreases, with focus change greater
        (100, 100, 150, 80, 50, -20, (0, 0, 30, 0, 20, 0)),
        # Test when focus increases and comparison decreases, with comparison change greater
        (100, 100, 110, 50, 10, -50, (0, 0, 0, 0, 10, 0)),
        # Test when both focus and comparison decrease, but focus decrease is greater
        (100, 100, 50, 80, -50, -20, (0, 0, 0, -50, 0, 0)),
        # Test when both focus and comparison decrease, but comparison decrease is greater
        (100, 100, 80, 50, -20, -50, (0, 0, 0, -20, 0, 0)),
        # Test when focus decreases and comparison increases, with focus change greater
        (100, 100, 50, 110, -50, 10, (0, 0, 0, -40, 0, -10)),
        # Test when focus decreases and comparison increases, with comparison change greater
        (100, 100, 80, 150, -20, 50, (0, 0, 0, 0, 0, -20)),
        # Test switch from comparison to focus, with focus change greater
        (100, 100, 180, 50, 80, -50, (0, 0, 30, 0, 50, 0)),
        # Test switch from comparison to focus, with comparison change greater
        (100, 100, 150, 20, 50, -80, (0, 0, 0, 0, 50, 0)),
        # Test switch from focus to comparison, with focus change greater
        (100, 100, 20, 150, -80, 50, (0, 0, 0, -30, 0, -50)),
        # Test switch from focus to comparison, with comparison change greater
        (100, 100, 50, 180, -50, 80, (0, 0, 0, 0, 0, -50)),
        # Test when there's no change in focus or comparison
        (100, 100, 100, 100, 0, 0, (0, 0, 0, 0, 0, 0)),
        # Test when a new customer is added (focus_p1 and comparison_p1 are 0)
        (0, 0, 1, 0, 1, 0, (1, 0, 0, 0, 0, 0)),
        # Test when a customer is lost (focus_p2 and comparison_p2 are 0)
        (1, 0, 0, 0, -1, 0, (0, -1, 0, 0, 0, 0)),
        # Test when both focus and comparison increase
        (0, 0, 1, 1, 1, 1, (1, 0, 0, 0, 0, 0)),
        # Test when focus increases and comparison decreases
        (0, 1, 1, 0, 1, -1, (0, 0, 0, 0, 1, 0)),
        # Test when both focus and comparison decrease
        (1, 1, 0, 0, -1, -1, (0, -1, 0, 0, 0, 0)),
        # Test when focus decreases and comparison increases
        (1, 0, 0, 1, -1, 1, (0, 0, 0, 0, 0, -1)),
        # Test when there's no change in focus or comparison
        (1, 1, 1, 1, 0, 0, (0, 0, 0, 0, 0, 0)),
    ],
)
def test_process_customer_group(
    focus_p1,
    comparison_p1,
    focus_p2,
    comparison_p2,
    focus_diff,
    comparison_diff,
    expected,
):
    """Test the process_customer_group method of the GainLoss class."""
    result = GainLoss.process_customer_group(
        focus_p1=focus_p1,
        comparison_p1=comparison_p1,
        focus_p2=focus_p2,
        comparison_p2=comparison_p2,
        focus_diff=focus_diff,
        comparison_diff=comparison_diff,
    )
    assert result == expected
