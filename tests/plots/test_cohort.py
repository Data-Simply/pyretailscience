"""Tests for the cohort plot module."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import cohort
from pyretailscience.style import graph_utils as gu

RNG = np.random.default_rng(42)


@pytest.fixture
def sample_cohort_dataframe():
    """Generates a sample cohort DataFrame."""
    data = np.round(RNG.uniform(0, 1, size=(6, 6)), 2)
    return pd.DataFrame(data, columns=[f"Month {i + 1}" for i in range(6)], index=[f"Cohort {i + 1}" for i in range(6)])


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mocks graph utility functions to avoid modifying global styles."""
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch(
        "pyretailscience.style.graph_utils.add_source_text",
        side_effect=lambda ax, source_text: ax,
    )


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort(sample_cohort_dataframe):
    """Test cohort plot with a standard DataFrame."""
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        x_label="Months",
        y_label="Cohorts",
        title="Cohort Retention Heatmap",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_source_text(sample_cohort_dataframe):
    """Test cohort plot with source text annotation."""
    source_text = "Source: Test Data"

    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_percentage(sample_cohort_dataframe):
    """Test cohort plot with percentage formatting enabled."""
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        percentage=True,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_ax_none(sample_cohort_dataframe):
    """Test cohort plot when ax is None (should create a new figure)."""
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        ax=None,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0
    assert result_ax.figure is not None


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_cohort_with_figsize(sample_cohort_dataframe):
    """Test cohort plot with a specified figsize."""
    width = 14
    height = 10
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        figsize=(width, height),
    )

    assert isinstance(result_ax, Axes)
    assert result_ax.figure.get_size_inches()[0] == width
    assert result_ax.figure.get_size_inches()[1] == height
