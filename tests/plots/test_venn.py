"""Tests for the plots.venn module."""

import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import venn
from pyretailscience.style import graph_utils as gu


@pytest.fixture
def sample_venn_dataframe():
    """A sample DataFrame for Venn diagram testing."""
    data = {
        "groups": [(1, 0), (0, 1), (1, 1)],
        "percent": [0.4, 0.3, 0.3],
    }
    return pd.DataFrame(data)


@pytest.fixture
def _mock_gu_functions(mocker):
    mocker.patch(
        "pyretailscience.style.graph_utils.add_source_text",
        side_effect=lambda ax, source_text, is_venn_diagram=False: ax,
    )


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_two_set_venn(sample_venn_dataframe):
    """Test Venn diagram plotting with two sets."""
    result_ax = venn.plot(
        df=sample_venn_dataframe,
        labels=["Set A", "Set B"],
        title="Test Venn Diagram",
    )
    assert isinstance(result_ax, Axes)
    assert len(result_ax.get_children()) > 0


@pytest.mark.usefixtures("_mock_gu_functions")
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


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_invalid_sets():
    """Test Venn plot with invalid number of sets (should raise ValueError)."""
    df = pd.DataFrame({"groups": [(1,)], "percent": [1.0]})
    with pytest.raises(ValueError, match="Only 2-set or 3-set Venn diagrams are supported."):
        venn.plot(df=df, labels=["Set A"])


@pytest.mark.usefixtures("_mock_gu_functions")
def test_plot_adds_source_text(sample_venn_dataframe):
    """Test Venn diagram adds source text."""
    source_text = "Source: Test Data"
    result_ax = venn.plot(
        df=sample_venn_dataframe,
        labels=["Set A", "Set B"],
        title="Test Venn Diagram with Source",
        source_text=source_text,
    )
    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text, is_venn_diagram=True)


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
