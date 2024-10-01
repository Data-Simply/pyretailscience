"""Tests for the standard_graphs module."""

from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from pyretailscience.standard_graphs import get_indexes, waterfall_plot
from pyretailscience.style.tailwind import COLORS, get_base_cmap


def test_get_indexes_single_column():
    """Test that the function works with a single column index."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [77.77777778, 100, 106.0606]})
    output = get_indexes(
        df=df,
        index_col="group_col",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_two_columns():
    """Test that the function works with two columns as the index."""
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
            "group_col2": ["D", "D", "D", "D", "D", "D", "E", "E", "E", "E", "E", "E"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col2": ["D", "D", "D", "E", "E", "E"],
            "group_col1": ["A", "B", "C", "A", "B", "C"],
            "index": [77.77777778, 100, 106.0606, 98.51851852, 100, 100.9661836],
        },
    )
    output = get_indexes(
        df=df,
        index_col="group_col1",
        index_subgroup_col="group_col2",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_offset():
    """Test that the function works with an offset parameter."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    expected_output = pd.DataFrame({"group_col": ["A", "B", "C"], "index": [-22.22222222, 0, 6.060606061]})
    output = get_indexes(
        df=df,
        index_col="group_col",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
        offset=100,
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_with_agg_func():
    """Test that the function works with the nunique agg_func parameter."""
    df = pd.DataFrame(
        {
            "group_col1": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "filter_col": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "value_col": [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 5, 8],
        },
    )
    expected_output = pd.DataFrame(
        {
            "group_col1": ["A", "B", "C"],
            "index": [140, 140, 46.6666667],
        },
    )
    output = get_indexes(
        df=df,
        index_col="group_col1",
        df_index_filter=df["filter_col"] == "X",
        value_col="value_col",
        agg_func="nunique",
    )
    pd.testing.assert_frame_equal(output, expected_output)


def test_get_indexes_index_filter_all_same():
    """Test that the function raises a ValueError when all the values in the index filter are the same."""
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B", "C", "C"],
            "filter_col": ["X", "X", "X", "X", "X", "X"],
            "value_col": [1, 2, 3, 4, 5, 6],
        },
    )
    # Assert a value error will be reaised
    with pytest.raises(ValueError):
        get_indexes(
            df=df,
            df_index_filter=[True, True, True, True, True, True],
            index_col="group_col",
            value_col="value_col",
        )

    with pytest.raises(ValueError):
        get_indexes(
            df=df,
            df_index_filter=[False, False, False, False, False, False],
            index_col="group_col",
            value_col="value_col",
        )


def test_get_base_cmap_three_or_fewer_colors():
    """Test the get_base_cmap function with three or fewer colors."""
    # Test with 3 colors (all green shades)
    gen = get_base_cmap(num_colors=3)
    expected_colors = [COLORS["green"][500], COLORS["green"][300], COLORS["green"][700]]

    # Get the first three colors from the generator
    generated_colors = list(islice(gen, 3))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors


def test_get_base_cmap_two_colors():
    """Test the get_base_cmap function with two colors."""
    # Test with 2 colors (only green 500 and green 300)
    gen = get_base_cmap(num_colors=2)
    expected_colors = [COLORS["green"][500], COLORS["green"][300]]

    # Get the first two colors from the generator
    generated_colors = list(islice(gen, 2))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors


def test_get_base_cmap_one_color():
    """Test the get_base_cmap function with one color."""
    # Test with 1 color (only green 500)
    gen = get_base_cmap(num_colors=1)
    expected_colors = [COLORS["green"][500]]

    # Get the first color from the generator
    generated_colors = list(islice(gen, 1))

    # Ensure the generated color matches the expected one
    assert generated_colors == expected_colors


def test_get_base_cmap_more_than_three_colors():
    """Test the get_base_cmap function with more than three colors."""
    # Test with 4 colors (mix of all colors)
    gen = get_base_cmap(num_colors=4)
    expected_colors = [
        COLORS["green"][500],
        COLORS["orange"][500],
        COLORS["red"][500],
        COLORS["blue"][500],
    ]

    # Get the first four colors from the generator
    generated_colors = list(islice(gen, 4))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors


def test_get_base_cmap_more_than_available_colors():
    """Test the get_base_cmap function with more colors than available."""
    # Test with more colors than available, ensure cycling occurs
    gen = get_base_cmap(num_colors=9)
    expected_colors = [
        COLORS["green"][500],
        COLORS["orange"][500],
        COLORS["red"][500],
        COLORS["blue"][500],
        COLORS["yellow"][500],
        COLORS["violet"][500],
        COLORS["pink"][500],
        COLORS["green"][300],
        COLORS["orange"][300],
    ]

    # Get the first nine colors from the generator
    generated_colors = list(islice(gen, 9))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors


def test_get_base_cmap_cycle_behavior():
    """Test the cycling behavior of the get_base_cmap function."""
    # Test with cycling colors
    gen = get_base_cmap(num_colors=3)

    # Get the first six colors (should cycle after the third)
    generated_colors = list(islice(gen, 6))
    expected_colors = [
        COLORS["green"][500],
        COLORS["green"][300],
        COLORS["green"][700],
        COLORS["green"][500],
        COLORS["green"][300],
        COLORS["green"][700],
    ]

    # Ensure the generated colors match the expected cycling behavior
    assert generated_colors == expected_colors


class TestWaterfallPlot:
    """Tests for the waterfall_plot function."""

    @pytest.fixture()
    def test_data(self):
        """Return a list of amounts."""
        return [100, -50, 30, -10], ["Start", "Decrease", "Increase", "End"]

    def test_generates_waterfall_plot_with_default_parameters(self, test_data):
        """Test that the function generates a waterfall plot with default parameters."""
        amounts, labels = test_data
        result_ax = waterfall_plot(amounts, labels)

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)

    def test_raises_value_error_for_mismatched_lengths(self, test_data):
        """Test that the function raises a ValueError when the lengths of amounts and labels are mismatched."""
        amounts, labels = test_data
        # Remove a value from amounts
        amounts.pop()

        with pytest.raises(ValueError, match="The lengths of amounts and labels must be the same."):
            waterfall_plot(amounts, labels)

    def test_generates_waterfall_plot_with_source_text(self, test_data):
        """Test that the function generates a waterfall plot with source text."""
        amounts, labels = test_data
        source_text = "Data source: Company XYZ"

        result_ax = waterfall_plot(amounts, labels, source_text=source_text)

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)

        # Check if the source text is added correctly
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_plot_colors_assigned_correctly_replicated_replicated(self, test_data):
        """Test that the function assigns colors correctly to the bars."""
        amounts, labels = test_data

        result_ax = waterfall_plot(amounts, labels)

        colors = [to_hex(patch.get_facecolor()) for patch in result_ax.patches]

        positive_color = COLORS["green"][500]
        negative_color = COLORS["red"][500]

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)
        assert colors == [
            positive_color,
            negative_color,
            positive_color,
            negative_color,
        ]

    def test_net_bar_colored_blue(self, test_data):
        """Test that the net bar is colored blue."""
        amounts, labels = test_data

        result_ax = waterfall_plot(amounts, labels, display_net_bar=True)

        last_bar_color = [to_hex(patch.get_facecolor()) for patch in result_ax.patches][-1]
        net_bar_blue = COLORS["blue"][500]

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels) + 1  # Check if 5 bars are plotted (including net bar)
        assert last_bar_color == net_bar_blue

    def test_generates_waterfall_plot_with_zero_amounts_removed(self, test_data):
        """Test that the function generates a waterfall plot with zero amounts removed."""
        amounts, labels = test_data
        # Set the first amount to zero
        amounts[0] = 0

        result_ax = waterfall_plot(amounts, labels)

        non_zero_amounts = len([amount for amount in amounts if amount != 0])

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == non_zero_amounts

    # Raises a ValueError for an invalid data label format
    def test_raises_value_error_for_invalid_data_label_format(self, test_data):
        """Test that the function raises a ValueError for an invalid data label format."""
        amounts, labels = test_data

        with pytest.raises(ValueError):
            waterfall_plot(amounts, labels, data_label_format="invalid_format")
