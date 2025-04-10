"""Tests for the GainLoss class in the gain_loss module."""

import datetime

import pandas as pd
import pytest
from matplotlib.axes import Axes

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


@pytest.fixture
def sample_df():
    """Sample transaction DataFrame for testing."""
    data = {
        "transaction_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "customer_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4],
        "transaction_date": [
            datetime.date(2023, 1, 15),
            datetime.date(2023, 1, 20),
            datetime.date(2023, 2, 5),
            datetime.date(2023, 2, 10),
            datetime.date(2023, 3, 1),
            datetime.date(2023, 3, 15),
            datetime.date(2023, 3, 20),
            datetime.date(2023, 4, 10),
            datetime.date(2023, 4, 25),
            datetime.date(2023, 5, 5),
            datetime.date(2023, 5, 20),
            datetime.date(2023, 6, 10),
        ],
        "unit_spend": [100, 150, 200, 120, 160, 210, 130, 170, 220, 140, 180, 230],
        "brand": [
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
        ],
    }
    return pd.DataFrame(data)


def test_overlap_in_p1_p2_raises(sample_df):
    """Test that GainLoss raises a ValueError when p1 and p2 indices overlap."""
    p1 = [True] * 6 + [False] * 6
    p2 = [True] + [False] * 11

    with pytest.raises(ValueError, match="p1_index and p2_index should not overlap"):
        GainLoss(
            df=sample_df,
            p1_index=p1,
            p2_index=p2,
            focus_group_index=[True] * 6 + [False] * 6,
            focus_group_name="Brand A",
            comparison_group_index=[False] * 6 + [True] * 6,
            comparison_group_name="Brand B",
        )


def test_overlap_in_focus_comparison_raises(sample_df):
    """Test that GainLoss raises a ValueError when focus and comparison indices overlap."""
    with pytest.raises(ValueError, match="focus_group_index and comparison_group_index should not overlap"):
        GainLoss(
            df=sample_df,
            p1_index=[True] * 6 + [False] * 6,
            p2_index=[False] * 6 + [True] * 6,
            focus_group_index=[True] * 6 + [False] * 6,
            focus_group_name="Brand A",
            comparison_group_index=[True] * 6 + [False] * 6,
            comparison_group_name="Brand B",
        )


def test_missing_required_column_raises():
    """Test that GainLoss raises a ValueError if required columns are missing in the DataFrame."""
    df = pd.DataFrame(
        {
            "spend": [100, 200],
            "group": ["A", "B"],
        },
    )
    with pytest.raises(ValueError, match="columns are required but missing"):
        GainLoss(
            df=df,
            p1_index=[True, False],
            p2_index=[False, True],
            focus_group_index=[True, False],
            focus_group_name="A",
            comparison_group_index=[False, True],
            comparison_group_name="B",
        )


def test_valid_gainloss(sample_df):
    """Test that a GainLoss object is successfully with valid input."""
    gl = GainLoss(
        df=sample_df,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )

    assert isinstance(gl, GainLoss)


def test_calc_gains_loss_table_without_group():
    """Test _calc_gains_loss_table with a flat DataFrame (no group index)."""
    df = pd.DataFrame(
        {
            "gain": [10, -5, 20],
            "loss": [5, 2, 1],
        },
    )
    result = GainLoss._calc_gains_loss_table(df)
    expected = pd.DataFrame(
        {
            "gain": [25],
            "loss": [8],
        },
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_calc_gains_loss_table_with_group():
    """Test _calc_gains_loss_table with grouped DataFrame."""
    idx = pd.MultiIndex.from_tuples([("Group A", 1), ("Group A", 2), ("Group B", 1)], names=["group", "id"])
    df = pd.DataFrame(
        {
            "gain": [10, 15, 5],
            "loss": [1, 2, 3],
        },
        index=idx,
    )

    result = GainLoss._calc_gains_loss_table(df, group_col="group")

    expected = pd.DataFrame(
        {
            "gain": [25, 5],
            "loss": [3, 3],
        },
        index=pd.Index(["Group A", "Group B"], name="group"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_plot_returns_figure_from_gainloss(sample_df):
    """Test that the `plot` method of the GainLoss class returns a valid matplotlib Axes object."""
    p1 = sample_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2 = sample_df["transaction_date"] >= datetime.date(2023, 5, 1)

    focus_group_index = sample_df["brand"] == "Brand A"
    comparison_group_index = sample_df["brand"] == "Brand B"

    gl = GainLoss(
        df=sample_df,
        p1_index=p1,
        p2_index=p2,
        focus_group_index=focus_group_index,
        focus_group_name="Brand A",
        comparison_group_index=comparison_group_index,
        comparison_group_name="Brand B",
    )

    fig = gl.plot()
    assert isinstance(fig, Axes)
