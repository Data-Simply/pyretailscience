"""Tests for the period_on_period overlapping_periods function."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.period_on_period import plot


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "value": range(20),
    }
    return pd.DataFrame(data)


def test_overlapping_periods_basic(sample_dataframe):
    """Test basic overlapping periods plot."""
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
        periods=periods,
    )

    assert isinstance(ax, Axes)
    expected_lines_count = 2
    assert len(ax.get_lines()) == expected_lines_count


def test_overlapping_periods_with_labels_and_title(sample_dataframe):
    """The plot renders the supplied title and axis labels on the axes."""
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    title = "Overlapping Periods Test"
    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
        periods=periods,
        x_label="Time",
        y_label="Sales",
        title=title,
    )

    assert ax.get_title() == title
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Sales"


def test_overlapping_periods_with_source_text(sample_dataframe):
    """The plot renders source_text as a figure-level text element."""
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    source_text = "Source: Sales Data"

    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
        periods=periods,
        source_text=source_text,
    )

    rendered = [t.get_text() for t in ax.figure.texts]
    assert source_text in rendered


def test_overlapping_periods_with_legend_title_and_outside(sample_dataframe):
    """move_legend_outside=True anchors the legend outside, with the supplied title."""
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    legend_title = "Periods"

    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
        periods=periods,
        move_legend_outside=True,
        legend_title=legend_title,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == legend_title
    anchor = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    expected_x, expected_y = PlotStyleHelper().legend_bbox_to_anchor
    assert anchor.x0 == pytest.approx(expected_x)
    assert anchor.y0 == pytest.approx(expected_y)


def test_overlapping_periods_raises_on_empty_periods(sample_dataframe):
    """Test overlapping periods raises a ValueError when an empty list is passed."""
    with pytest.raises(
        ValueError,
        match=r"The 'periods' list must contain at least two \(start, end\) tuples for comparison",
    ):
        plot(
            df=sample_dataframe,
            x_col="date",
            value_col="value",
            periods=[],
        )
