"""Tests for the bar plot module."""

from itertools import cycle

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from pyretailscience.plots import bar
from pyretailscience.style import graph_utils as gu


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing."""
    data = {
        "product": ["A", "B", "C", "D"],
        "sales_q1": [1000, 1500, 2000, 2500],
        "sales_q2": [1100, 1600, 2100, 2600],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_series():
    """A sample series for testing."""
    return pd.Series([1000, 1500, 2000, 2500], index=["A", "B", "C", "D"])


@pytest.fixture
def _mock_color_generators(mocker):
    """Mock the color generators for single and multi color maps."""
    single_color_gen = cycle(["#FF0000"])  # Mocked single-color generator (e.g., red)
    multi_color_gen = cycle(["#FF0000", "#00FF00", "#0000FF"])  # Mocked multi-color generator
    mocker.patch("pyretailscience.style.tailwind.get_single_color_cmap", return_value=single_color_gen)
    mocker.patch("pyretailscience.style.tailwind.get_multi_color_cmap", return_value=multi_color_gen)


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mock the standard graph utilities functions."""
    mocker.patch("pyretailscience.style.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.style.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch("pyretailscience.style.graph_utils.add_source_text", side_effect=lambda ax, source_text: ax)
    mocker.patch("pyretailscience.style.graph_utils.apply_hatches", side_effect=lambda ax, num_segments: ax)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_empty_dataframe():
    """Test bar plot with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["sales_q1", "sales_q2"])

    with pytest.raises(ValueError, match="Cannot plot with empty DataFrame"):
        bar.plot(
            df=empty_df,
            value_col="sales_q1",
            x_col="product",
            title="Test Plot with Empty DataFrame",
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_single_bar(sample_dataframe):
    """Test the bar plot function with a single value column."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        title="Test Single Bar Plot",
    )

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_missing_x_col_in_dataframe(sample_dataframe):
    """Test bar plot when the provided x_col does not exist in the DataFrame."""
    with pytest.raises(KeyError, match="x_col 'missing_col' not found in DataFrame"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales_q1",
            x_col="missing_col",
            title="Test Plot with Missing x_col",
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_none_value_col(sample_dataframe):
    """Test bar plot with None for value_col (default 'Value' column)."""
    result_ax = bar.plot(
        df=sample_dataframe["sales_q1"],
        title="Test Plot with None Value Col",
    )

    expected_heights = sample_dataframe["sales_q1"].tolist()
    actual_heights = [p.get_height() for p in result_ax.patches]

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert actual_heights == pytest.approx(expected_heights)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_nan_values():
    """Test bar plot with NaN values in the data."""
    nan_dataframe = pd.DataFrame(
        {
            "product": ["A", "B", "C", "D"],
            "sales_q1": [1000, 1500, None, 2500],
            "sales_q2": [1100, 1600, 2100, None],
        },
    )

    result_ax = bar.plot(
        df=nan_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        data_label_format="absolute",
        title="Test Plot with NaN Values",
    )

    expected_heights = [1000.0, 1500.0, 0.0, 2500.0, 1100.0, 1600.0, 2100.0, 0.0]
    actual_heights = [p.get_height() for p in result_ax.patches]

    expected_num_patches = 8

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert actual_heights == pytest.approx(expected_heights)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_horizontal_orientation(sample_dataframe):
    """Test bar plot with horizontal orientation."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        orientation="horizontal",
        title="Test Horizontal Bar Plot",
    )

    expected_height = 0.8

    assert isinstance(result_ax, Axes)
    # Check that the bars are oriented horizontally by checking their width, not height
    assert all(p.get_width() > 0 for p in result_ax.patches)
    assert all(p.get_height() == expected_height for p in result_ax.patches)  # Default bar height in horizontal bars


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_vertical_orientation(sample_dataframe):
    """Test bar plot with vertical orientation."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        orientation="vertical",
        title="Test Vertical Bar Plot",
    )

    expected_width = 0.8

    assert isinstance(result_ax, Axes)
    # Check that the bars are oriented vertically by checking their height, not width
    assert all(p.get_height() > 0 for p in result_ax.patches)
    assert all(p.get_width() == expected_width for p in result_ax.patches)  # Default bar width in vertical bars


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_custom_bar_width(sample_dataframe):
    """Test bar plot with a custom width for bars."""
    custom_width = 0.5
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        width=custom_width,
        title="Test Plot with Custom Bar Width",
    )

    assert isinstance(result_ax, Axes)
    # Check the width of individual bars
    assert all(p.get_width() == custom_width for p in result_ax.patches if isinstance(p, Rectangle))


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_legend_outside(sample_dataframe):
    """Test the bar plot function moves the legend outside the plot."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
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
def test_plot_with_hatch(sample_dataframe):
    """Test the bar plot function with hatching."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        use_hatch=True,
        title="Test Bar Plot with Hatch",
    )

    gu.apply_hatches.assert_called_once_with(ax=result_ax, num_segments=1)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_multiple_bars(sample_dataframe):
    """Test the bar plot function with multiple value columns."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        title="Test Multiple Bar Plot",
    )

    expected_heights = sample_dataframe["sales_q1"].tolist() + sample_dataframe["sales_q2"].tolist()
    actual_heights = [p.get_height() for p in result_ax.patches]

    expected_num_patches = 8

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert actual_heights == pytest.approx(expected_heights)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_sorting(sample_dataframe):
    """Test the bar plot function with sorting in ascending order."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        sort_order="ascending",
        title="Test Sorted Bar Plot",
    )

    expected_order = ["A", "B", "C", "D"]
    actual_order = [label.get_text() for label in result_ax.get_xticklabels()]
    bar_heights = [patch.get_height() for patch in result_ax.patches]
    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches
    assert result_ax.get_xticklabels()[0].get_text() == "A"
    assert actual_order == expected_order
    assert bar_heights == sorted(bar_heights)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_data_labels(sample_dataframe):
    """Test the bar plot function with data labels in absolute format."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        data_label_format="absolute",
        title="Test Bar Plot with Data Labels",
    )

    expected_labels = ["1K", "1.5K", "2K", "2.5K"]
    actual_labels = [text.get_text() for text in result_ax.texts]

    assert isinstance(result_ax, Axes)
    assert len(result_ax.containers) == 1
    assert actual_labels == expected_labels


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_percentage_by_bar_group_labels(sample_dataframe):
    """Test the bar plot function with data labels in 'percentage_by_bar_group' format and verify percentages."""
    # Plot the bars with percentage labels
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        data_label_format="percentage_by_bar_group",
        title="Test Bar Plot with Percentage Within Bar Groups Labels",
    )

    # Calculate expected percentages within each bar group (sum within each row)
    total_sales_per_product = sample_dataframe["sales_q1"] + sample_dataframe["sales_q2"]
    expected_percentages_q1 = (sample_dataframe["sales_q1"] / total_sales_per_product) * 100
    expected_percentages_q2 = (sample_dataframe["sales_q2"] / total_sales_per_product) * 100

    # Combine the expected percentages for both Q1 and Q2
    expected_percentages = list(expected_percentages_q1) + list(expected_percentages_q2)

    # Retrieve all text labels applied to the bars
    labels = [t.get_text() for t in result_ax.texts]

    # Check that the retrieved labels match the expected percentages
    for label, expected_percentage in zip(labels, expected_percentages, strict=False):
        assert float(label.strip("%")) == pytest.approx(expected_percentage, 0.01)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_percentage_by_series_labels(sample_dataframe):
    """Test the bar plot function with data labels in 'percentage_by_series' format and verify percentages."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        x_col="product",
        data_label_format="percentage_by_series",
        title="Test Bar Plot with Percentage Across Bar Groups Labels",
    )

    # Calculate expected percentages across bar groups for each value column
    total_sum_q1 = sample_dataframe["sales_q1"].sum()
    total_sum_q2 = sample_dataframe["sales_q2"].sum()

    expected_percentages_q1 = (sample_dataframe["sales_q1"] / total_sum_q1) * 100
    expected_percentages_q2 = (sample_dataframe["sales_q2"] / total_sum_q2) * 100

    # Combine the expected percentages for both Q1 and Q2
    expected_percentages = list(expected_percentages_q1) + list(expected_percentages_q2)

    # Retrieve all text labels applied to the bars
    labels = [t.get_text() for t in result_ax.texts]

    # Check that the retrieved labels match the expected percentages
    for label, expected_percentage in zip(labels, expected_percentages, strict=False):
        assert float(label.strip("%")) == pytest.approx(expected_percentage, 0.29)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_adds_source_text(sample_dataframe):
    """Test the bar plot function adds source text to the plot."""
    source_text = "Source: Test Data"
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
        title="Test Bar Plot with Source Text",
        source_text=source_text,
    )

    gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_calls_dataframe_plot(mocker, sample_dataframe):
    """Test that pandas.DataFrame.plot is called with correct arguments."""
    # Spy on DataFrame.plot
    mock_df_plot = mocker.patch("pandas.DataFrame.plot")

    # Call the bar plot function
    bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        x_col="product",
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
        width=0.8,
    )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_invalid_orientation(sample_dataframe):
    """Test that an invalid orientation raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid orientation: invalid_orientation. Expected one of .*"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            x_col="product",
            orientation="invalid_orientation",  # Invalid value
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_invalid_sort_order(sample_dataframe):
    """Test that an invalid sort_order raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid sort_order: invalid_sort_order. Expected one of .*"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            x_col="product",
            sort_order="invalid_sort_order",  # Invalid value
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_invalid_data_label_format(sample_dataframe):
    """Test that an invalid data_label_format raises a ValueError."""
    with pytest.raises(ValueError, match="Invalid data_label_format: invalid_format. Expected one of .*"):
        bar.plot(
            df=sample_dataframe,
            value_col="sales",
            x_col="product",
            data_label_format="invalid_format",  # Invalid value
        )


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_default_value_col_handling(sample_series):
    """Test that a default 'Value' column is created when no value_col is passed."""
    result_ax = bar.plot(
        df=sample_series,
        title="Test Default Value Column Handling",
    )

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_with_series(sample_series):
    """Test the bar plot function works with a pandas Series."""
    result_ax = bar.plot(
        df=sample_series,
        title="Test Bar Plot with Series",
    )

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_no_x_col_string_value_col(sample_dataframe):
    """Test bar plot when x_col is None and value_col is a string."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col="sales_q1",
        title="Test Plot: No Group Col and String Value Col",
    )

    expected_num_patches = 4

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_plot_no_x_col_list_value_col(sample_dataframe):
    """Test bar plot when x_col is None and value_col is a list."""
    result_ax = bar.plot(
        df=sample_dataframe,
        value_col=["sales_q1", "sales_q2"],
        title="Test Plot: No Group Col and List Value Col",
    )

    expected_num_patches = 8

    assert isinstance(result_ax, Axes)
    assert len(result_ax.patches) == expected_num_patches


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_percentage_by_bar_group_with_negative_values():
    """Test percentage_by_bar_group with negative values, triggering warning."""
    df = pd.DataFrame(
        {
            "product": ["A", "B", "C", "D"],
            "sales_q1": [1000, -1500, 2000, -2500],
            "sales_q2": [-1100, 1600, -2100, 2600],
        },
    )
    with pytest.warns(UserWarning, match="Negative values detected"):
        bar.plot(
            df=df,
            value_col=["sales_q1", "sales_q2"],
            x_col="product",
            data_label_format="percentage_by_bar_group",
        )
    plt.close("all")


@pytest.mark.usefixtures("_mock_color_generators", "_mock_gu_functions")
def test_percentage_by_bar_group_with_zero_group_total():
    """Test percentage_by_bar_group with zero group totals and suppress the warning."""
    df = pd.DataFrame({"product": ["A", "B"], "sales": [0, 0]})

    with pytest.warns(UserWarning, match="Division by zero detected"):
        result_ax = bar.plot(
            df=df,
            value_col="sales",
            x_col="product",
            data_label_format="percentage_by_bar_group",
        )

    labels = [t.get_text() for t in result_ax.texts]
    assert all(label == "" for label in labels)  # Should all be empty due to division by zero
    plt.close("all")
