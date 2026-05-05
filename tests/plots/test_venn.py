"""Tests for the plots.venn module."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.plots import venn


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_venn_dataframe():
    """A sample DataFrame for Venn diagram testing."""
    data = {
        "groups": [(1, 0), (0, 1), (1, 1)],
        "percent": [0.4, 0.3, 0.3],
    }
    return pd.DataFrame(data)


def test_plot_two_set_venn(sample_venn_dataframe):
    """Test Venn diagram plotting with two sets."""
    result_ax = venn.plot(
        df=sample_venn_dataframe,
        labels=["Set A", "Set B"],
        title="Test Venn Diagram",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


def test_plot_three_set_venn():
    """Test Venn diagram plotting with three sets."""
    df = pd.DataFrame(
        {
            "groups": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
            "percent": [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
        },
    )
    result_ax = venn.plot(
        df=df,
        labels=["Set A", "Set B", "Set C"],
        title="Three-set Venn Diagram",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


def test_plot_invalid_sets():
    """Test Venn plot with invalid number of sets (should raise ValueError)."""
    df = pd.DataFrame({"groups": [(1,)], "percent": [1.0]})
    with pytest.raises(ValueError, match="Only 2-set or 3-set Venn diagrams are supported"):
        venn.plot(df=df, labels=["Set A"])


def test_plot_adds_source_text(sample_venn_dataframe):
    """The Venn diagram renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"
    result_ax = venn.plot(
        df=sample_venn_dataframe,
        labels=["Set A", "Set B"],
        title="Test Venn Diagram with Source",
        source_text=source_text,
    )
    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_venn_default_ax(sample_venn_dataframe):
    """Test Venn diagram when ax is None to ensure a new figure is created."""
    result_ax = venn.plot(df=sample_venn_dataframe, labels=["A", "B"])
    assert isinstance(result_ax, Axes)


def test_venn_with_title(sample_venn_dataframe):
    """Test Venn diagram with a title to cover ax.set_title."""
    title = "Test Venn Diagram"
    result_ax = venn.plot(df=sample_venn_dataframe, labels=["A", "B"], title=title)

    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() == title
