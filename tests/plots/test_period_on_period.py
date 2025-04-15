"""Tests for the period_on_period overlapping_periods function."""

from itertools import cycle

import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots.period_on_period import plot
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
    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
        periods=periods,
    )

    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) == EXPECTED_LINES_COUNT


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_overlapping_periods_with_labels_and_title(sample_dataframe):
    """Test overlapping periods with axis labels and title."""
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
    periods = [("2023-01-01", "2023-01-05"), ("2023-01-06", "2023-01-10")]
    source_text = "Source: Sales Data"

    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
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

    ax = plot(
        df=sample_dataframe,
        x_col="date",
        value_col="value",
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
def test_overlapping_periods_raises_on_empty_periods(sample_dataframe):
    """Test overlapping periods with a ValueError is raised if an empty list of periods is passed."""
    with pytest.raises(
        ValueError,
        match=r"The 'periods' list must contain at least two \(start, end\) tuples for comparison\.",
    ):
        plot(
            df=sample_dataframe,
            x_col="date",
            value_col="value",
            periods=[],
        )
