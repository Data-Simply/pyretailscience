"""Tests for the waterfall plot module."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from pyretailscience.plots.waterfall import plot
from pyretailscience.style.tailwind import COLORS


class TestWaterfallPlot:
    """Tests for the waterfall_plot function."""

    @pytest.fixture
    def test_data(self):
        """Return a list of amounts."""
        return [100, -50, 30, -10], ["Start", "Decrease", "Increase", "End"]

    def test_generates_waterfall_plot_with_default_parameters(self, test_data):
        """Test that the function generates a waterfall plot with default parameters."""
        amounts, labels = test_data
        result_ax = plot(amounts, labels)

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)

    def test_raises_value_error_for_mismatched_lengths(self, test_data):
        """Test that the function raises a ValueError when the lengths of amounts and labels are mismatched."""
        amounts, labels = test_data
        # Remove a value from amounts
        amounts.pop()

        with pytest.raises(
            ValueError,
            match="The lengths of amounts and labels must be the same.",
        ):
            plot(amounts, labels)

    def test_generates_waterfall_plot_with_source_text(self, test_data):
        """Test that the function generates a waterfall plot with source text."""
        amounts, labels = test_data
        source_text = "Data source: Company XYZ"

        result_ax = plot(amounts, labels, source_text=source_text)

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == len(labels)

        # Check if the source text is added correctly
        source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
        assert len(source_texts) == 1

    def test_plot_colors_assigned_correctly_replicated_replicated(self, test_data):
        """Test that the function assigns colors correctly to the bars."""
        amounts, labels = test_data

        result_ax = plot(amounts, labels)

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

        result_ax = plot(amounts, labels, display_net_bar=True)

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

        result_ax = plot(amounts, labels)

        non_zero_amounts = len([amount for amount in amounts if amount != 0])

        assert isinstance(result_ax, plt.Axes)
        assert len(result_ax.patches) == non_zero_amounts

    # Raises a ValueError for an invalid data label format
    def test_raises_value_error_for_invalid_data_label_format(self, test_data):
        """Test that the function raises a ValueError for an invalid data label format."""
        amounts, labels = test_data

        with pytest.raises(ValueError):
            plot(amounts, labels, data_label_format="invalid_format")
