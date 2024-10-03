"""Tests for the bar plot module."""

from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import bar
from pyretailscience.style import graph_utils as gu


@pytest.fixture()
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "product": ["A", "B", "C", "D"],
        "sales_q1": [1000, 1500, 2000, 2500],
        "sales_q2": [1100, 1600, 2100, 2600],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_series():
    """A sample series for testing."""
    return pd.Series([1000, 1500, 2000, 2500], index=["A", "B", "C", "D"])


@pytest.fixture()
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi color maps."""
    single_color_gen = cycle(["#FF0000"])  # Mocked single-color generator (e.g., red)
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])  # Mocked multi-color generator
    mocker.patch("pyretailscience.style.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.style.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture()
def _mock_gu_functions(mocker):
    """Mock the standard graph utilities functions."""
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)
    mocker.patch("pyretailscience.style.graph_utils.apply_hatches", side_effect=lambda ax, num_segments: ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_bar(sample_dataframe):
    """Test the bar plot function with a single value column."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        ax=ax,
        title="Test Single Bar Plot",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_bars(sample_dataframe):
    """Test the bar plot function with multiple value columns."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        group_col="product",
        ax=ax,
        title="Test Multiple Bar Plot",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that some bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_sorting(sample_dataframe):
    """Test the bar plot function with sorting in ascending order."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        sort_order="ascending",
        ax=ax,
        title="Test Sorted Bar Plot",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0
    assert result_ax.get_xticklabels()[0].get_text() == "A"  # Ensure the sorting was applied


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_hatch(sample_dataframe):
    """Test the bar plot function with hatching."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        use_hatch=True,
        ax=ax,
        title="Test Bar Plot with Hatch",
    )

    gu.apply_hatches.assert_called_once_with(ax=result_ax, num_segments=1)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_data_labels(sample_dataframe):
    """Test the bar plot function with data labels in absolute format."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        data_label_format="absolute",
        ax=ax,
        title="Test Bar Plot with Data Labels",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.containers) > 0  # Ensure that bars were plotted with labels


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_percentage_labels(sample_dataframe):
    """Test the bar plot function with data labels in percentage format."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        data_label_format="percentage",
        ax=ax,
        title="Test Bar Plot with Percentage Labels",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.containers) > 0  # Ensure that bars were plotted with percentage labels


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(sample_dataframe):
    """Test the bar plot function adds source text to the plot."""
    _, ax = plt.subplots()

    source_text = "Source: Test Data"
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        ax=ax,
        title="Test Bar Plot with Source Text",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_legend_outside(sample_dataframe):
    """Test the bar plot function moves the legend outside the plot."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        ax=ax,
        title="Test Bar Plot with Legend Outside",
        move_legend_outside=True,
    )

    gu.standard_graph_styles.assert_called_once_with(
        ax=result_ax,
        title="Test Bar Plot with Legend Outside",
        x_label=None,
        y_label=None,
        legend_title=None,
        move_legend_outside=True,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_calls_dataframe_plot(mocker, sample_dataframe):
    """Test that pandas.DataFrame.plot is called with correct arguments."""
    # Spy on DataFrame.plot
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    _, ax = plt.subplots()

    # Call the bar plot function
    bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        group_col="product",
        ax=ax,
        title="Test Bar Plot",
    )

    # Check the arguments passed to DataFrame.plot
    mock_df_plot.assert_called_once_with(
        kind="bar",
        y=["sales_q1"],
        x="product",
        ax=mocker.ANY,
        color=mocker.ANY,
        legend=False,
        width=0.35,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_invalid_orientation(sample_dataframe):
    """Test that an invalid orientation raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid orientation: invalid_orientation. Expected one of .*"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            group_col="category",
            orientation="invalid_orientation",  # Invalid value
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_invalid_sort_order(sample_dataframe):
    """Test that an invalid sort_order raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid sort_order: invalid_sort_order. Expected one of .*"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            group_col="category",
            sort_order="invalid_sort_order",  # Invalid value
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_invalid_data_label_format(sample_dataframe):
    """Test that an invalid data_label_format raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid data_label_format: invalid_format. Expected one of .*"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            group_col="category",
            data_label_format="invalid_format",  # Invalid value
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_default_value_col_handling(sample_series):
    """Test that a default 'Value' column is created when no value_col is passed."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_series,
        ax=ax,
        title="Test Default Value Column Handling",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_series(sample_series):
    """Test the bar plot function works with a pandas Series."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_series,
        ax=ax,
        title="Test Bar Plot with Series",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0  # Ensure that bars were plotted


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_no_group_col_string_value_col(sample_dataframe):
    """Test bar plot when group_col is None and value_col is a string."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        ax=ax,
        title="Test Plot: No Group Col and String Value Col",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_no_group_col_list_value_col(sample_dataframe):
    """Test bar plot when group_col is None and value_col is a list."""
    _, ax = plt.subplots()

    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        ax=ax,
        title="Test Plot: No Group Col and List Value Col",
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) > 0
