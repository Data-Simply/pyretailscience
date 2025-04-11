"""Tests for the period_on_period overlapping_periods function."""

import warnings
from itertools import cycle

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pyretailscience.plots.period_on_period import overlapping_periods
from pyretailscience.style import graph_utils as gu

EXPECTED_LINES_COUNT = 3


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "value": range(20),
    }
    return pd.DataFrame(data)


@pytest.fixture
def _mock_color_generators(mocker):
    """Mock single color generator."""
    mocker.patch("pyretailscience.style.tailwind.get_single_color_cmap", return_value=cycle(["#FF0000"]))


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mock graph utility functions."""
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_basic(sample_dataframe):
    """Test basic overlapping periods plot."""
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    ax = overlapping_periods(
        df=sample_dataframe,
        x_col="date",
        y_col="value",
        periods=periods,
    )

    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) == EXPECTED_LINES_COUNT


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_with_labels_and_title(sample_dataframe):
    """Test overlapping periods with axis labels and title."""
    periods = [("2023-01-01", "2023-01-05")]
    title = "Overlapping Periods Test"
    ax = overlapping_periods(
        df=sample_dataframe,
        x_col="date",
        y_col="value",
        periods=periods,
        x_label="Time",
        y_label="Sales",
        title=title,
    )

    assert isinstance(ax, Axes)
    gu.standard_graph_styles.assert_any_call(
        ax=ax,
        title=title,
        x_label="Time",
        y_label="Sales",
        legend_title=None,
        move_legend_outside=False,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_with_source_text(sample_dataframe):
    """Test overlapping periods with source text added."""
    periods = [("2023-01-01", "2023-01-05")]
    source_text = "Source: Sales Data"

    ax = overlapping_periods(
        df=sample_dataframe,
        x_col="date",
        y_col="value",
        periods=periods,
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=ax, source_text=source_text)
    assert isinstance(ax, Axes)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_with_legend_title_and_outside(sample_dataframe):
    """Test overlapping periods with legend title and moved legend."""
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    legend_title = "Periods"

    ax = overlapping_periods(
        df=sample_dataframe,
        x_col="date",
        y_col="value",
        periods=periods,
        move_legend_outside=True,
        legend_title=legend_title,
    )

    gu.standard_graph_styles.assert_any_call(
        ax=ax,
        title=None,
        x_label="date",
        y_label="value",
        legend_title=legend_title,
        move_legend_outside=True,
    )
    assert isinstance(ax, Axes)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_datetime_index_warns(sample_dataframe, mocker):
    """Test overlapping periods with a datetime index and no x_col, expecting a warning."""
    mocker.patch("warnings.warn")

    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    ax = overlapping_periods(
        df=sample_dataframe,
        x_col="date",
        y_col="value",
        periods=periods,
    )

    assert isinstance(ax, Axes)

    warnings.warn.assert_any_call(
        "The column 'realigned_date' is datetime-like. Consider using the 'plots.time_line' module for time-based plots.",
        UserWarning,
        stacklevel=2,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_raises_on_empty_periods(sample_dataframe):
    """Test overlapping periods with a ValueError is raised if an empty list of periods is passed."""
    with pytest.raises(ValueError, match="The 'periods' list must contain at least one"):
        overlapping_periods(
            df=sample_dataframe,
            x_col="date",
            y_col="value",
            periods=[],
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_skips_empty_period_df(sample_dataframe):
    """Test that periods with no matching data do not add any lines to the plot."""
    fig, ax = plt.subplots()
    initial_lines = len(ax.lines)

    periods = [("1900-01-01", "1900-01-05")]

    returned_ax = overlapping_periods(
        df=sample_dataframe,
        x_col="date",
        y_col="value",
        periods=periods,
        title="Empty Period Test",
        ax=ax,
    )

    assert isinstance(returned_ax, plt.Axes)
    assert len(returned_ax.lines) == initial_lines, (
        f"Expected {initial_lines} lines, but got {len(returned_ax.lines)}. "
        "New lines were added even though the period had no matching data."
    )
