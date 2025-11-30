"""Tests for the color module in the style package."""

import pytest

from pyretailscience.options import get_option, option_context
from pyretailscience.plots.styles.colors import (
    get_heatmap_cmap,
    get_named_color,
    get_plot_colors,
)


class TestGetPlotColors:
    """Test get_plot_colors() function behavior."""

    @pytest.mark.parametrize("num_series", [1, 3, 4, 10])
    def test_get_plot_colors_returns_correct_length(self, num_series):
        """Test that get_plot_colors returns correct number of colors."""
        colors = get_plot_colors(num_series)
        assert len(colors) == num_series

    @pytest.mark.parametrize(
        ("num_series", "expected_palette"),
        [
            (1, "mono"),  # 1 series: use monochromatic
            (3, "mono"),  # 3 series: use monochromatic (equal to default mono length)
            (4, "multi"),  # 4 series: switch to multi-color
            (10, "multi"),  # Many series: use multi-color
        ],
    )
    def test_get_plot_colors_palette_selection(self, num_series, expected_palette):
        """Test get_plot_colors() returns correct number of colors from appropriate palette."""
        colors = get_plot_colors(num_series)
        assert len(colors) == num_series

        mono_palette = get_option("plot.color.mono_palette")
        multi_palette = get_option("plot.color.multi_color_palette")

        if expected_palette == "mono":
            # All returned colors should be from monochromatic palette
            assert all(c in mono_palette for c in colors)
        else:
            # Colors should be from multi-color palette
            assert all(c in multi_palette for c in colors)

    def test_get_plot_colors_custom_mono_palette(self):
        """Test get_plot_colors() uses custom monochromatic palette."""
        custom_mono = ["#1e40af", "#93c5fd"]  # Only 2 colors
        two_series = 2
        three_series = 3
        with option_context("plot.color.mono_palette", custom_mono):
            # With 2 series, should use mono palette
            colors = get_plot_colors(two_series)
            assert len(colors) == two_series
            assert all(c in custom_mono for c in colors)

            # With 3 series, should switch to multi-color palette
            colors = get_plot_colors(three_series)
            assert len(colors) == three_series
            multi_palette = get_option("plot.color.multi_color_palette")
            assert all(c in multi_palette for c in colors)

    def test_get_plot_colors_empty_mono_palette(self):
        """Test get_plot_colors() with disabled monochromatic palette."""
        with option_context("plot.color.mono_palette", []):
            # Even with 1 series, should use multi-color palette
            colors = get_plot_colors(1)
            assert len(colors) == 1
            multi_palette = get_option("plot.color.multi_color_palette")
            assert colors[0] == multi_palette[0]

    def test_get_plot_colors_cycling_behavior(self):
        """Test get_plot_colors() cycles through multi-color palette when needed."""
        five_series = 5
        with option_context(
            "plot.color.mono_palette",
            [],  # Force multi-color
            "plot.color.multi_color_palette",
            ["#red", "#blue"],  # Only 2 colors
        ):
            colors = get_plot_colors(five_series)
            assert len(colors) == five_series
            # Should cycle: red, blue, red, blue, red
            expected = ["#red", "#blue", "#red", "#blue", "#red"]
            assert colors == expected


class TestGetNamedColor:
    """Test get_named_color() function behavior."""

    def test_get_named_color_custom_values(self):
        """Test get_named_color() uses custom configured values."""
        custom_positive = "#16a34a"  # green-600
        custom_negative = "#dc2626"  # red-600

        with option_context(
            "plot.color.positive",
            custom_positive,
            "plot.color.negative",
            custom_negative,
        ):
            assert get_named_color("positive") == custom_positive
            assert get_named_color("negative") == custom_negative

    def test_get_named_color_invalid_type(self):
        """Test get_named_color() raises error for invalid color type."""
        with pytest.raises(ValueError):
            get_named_color("invalid_color_type")


class TestGetHeatmapCmap:
    """Test get_heatmap_cmap() function behavior."""

    @pytest.mark.parametrize(
        ("cmap_config", "expected_type"),
        [
            ("green", "ListedColormap"),  # Tailwind color name
            ("blue", "ListedColormap"),  # Tailwind color name
            ("viridis", "ListedColormap"),  # Matplotlib colormap (ListedColormap in this version)
            ("Greens", "LinearSegmentedColormap"),  # Matplotlib colormap
        ],
    )
    def test_get_heatmap_cmap_supports_both_tailwind_and_matplotlib(self, cmap_config, expected_type):
        """Test get_heatmap_cmap() handles both Tailwind and matplotlib colormap names."""
        with option_context("plot.color.heatmap", cmap_config):
            cmap = get_heatmap_cmap()
            assert type(cmap).__name__ == expected_type
