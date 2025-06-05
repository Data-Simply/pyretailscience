"""Tests for the PlotStyler module."""

from unittest.mock import Mock

import pytest
from matplotlib import pyplot as plt

from pyretailscience.plots.styles.styling_helpers import PlotStyler


class TestPlotStyler:
    """Unit tests for PlotStyler class."""

    @pytest.fixture
    def fig_ax(self):
        """Fixture to create and yield a matplotlib figure and axis."""
        fig, ax = plt.subplots()
        yield fig, ax
        plt.close(fig)

    @pytest.fixture
    def styler(self):
        """Fixture to provide a PlotStyler instance."""
        return PlotStyler()

    def test_apply_base_styling(self, fig_ax, styler):
        """Test base styling application."""
        _, ax = fig_ax
        styler.apply_base_styling(ax)

        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.get_axisbelow()

    def test_apply_title(self, fig_ax, styler):
        """Test title styling application."""
        _, ax = fig_ax
        styler.apply_title(ax, "Test Title", pad=15)

        assert ax.get_title() == "Test Title"
        assert ax.title.get_fontsize() == styler.context.fonts.title_size

    @pytest.mark.parametrize(
        ("label_text", "axis", "get_label_func"),
        [
            ("X Label", "x", lambda ax: ax.get_xlabel()),
            ("Y Label", "y", lambda ax: ax.get_ylabel()),
        ],
    )
    def test_apply_label_axis(self, fig_ax, styler, label_text, axis, get_label_func):
        """Test axis label styling for both x and y axes."""
        _, ax = fig_ax
        styler.apply_label(ax, label_text, axis, pad=10)

        assert get_label_func(ax) == label_text
        label_obj = ax.xaxis.label if axis == "x" else ax.yaxis.label
        assert label_obj.get_fontsize() == styler.context.fonts.label_size

    def test_apply_ticks(self, fig_ax, styler):
        """Test tick styling application."""
        _, ax = fig_ax
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # add data to generate ticks
        styler.apply_ticks(ax)

        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        assert len(tick_labels) > 0
        for label in tick_labels:
            assert label.get_fontsize() == styler.context.fonts.tick_size

    def test_apply_source_text(self, fig_ax, styler):
        """Test source text styling."""
        _, ax = fig_ax
        source_text = styler.apply_source_text(ax, "Test Source", x=0.01, y=0.02)

        assert source_text.get_text() == "Test Source"
        assert source_text.get_color() == "dimgray"
        assert source_text.get_fontsize() == styler.context.fonts.source_size

    @pytest.mark.parametrize(
        ("outside", "expected_title"),
        [
            (False, "Test Legend"),
            (True, None),
        ],
    )
    def test_apply_legend(self, fig_ax, styler, outside, expected_title):
        """Test legend styling (inside and outside positioning)."""
        _, ax = fig_ax
        ax.plot([1, 2, 3], [1, 2, 3], label="Series 1")
        ax.plot([1, 2, 3], [3, 2, 1], label="Series 2")

        kwargs = {"outside": outside}
        if not outside:
            kwargs["title"] = expected_title

        styler.apply_legend(ax, **kwargs)

        legend = ax.get_legend()
        assert legend is not None

        if not outside:
            assert not legend.get_frame_on()
            assert legend.get_title().get_text() == expected_title
        else:
            assert legend.get_bbox_to_anchor() is not None

    def test_get_colors_integration_with_actual_plot(self, fig_ax, styler):
        """Integration test: verify colors work with actual matplotlib plot."""
        _, ax = fig_ax

        colors = styler.get_colors_for_plot(3)
        expected_colors = set(colors)

        x_data = [1, 2, 3]
        for i, color in enumerate(colors):
            ax.plot(x_data, [i + 1, i + 2, i + 3], color=color, label=f"Series {i + 1}")

        lines = ax.get_lines()
        expected_line = 3
        assert len(lines) == expected_line

        for i, line in enumerate(lines):
            actual_color = line.get_color()
            assert actual_color in expected_colors, (
                f"Line {i} color {actual_color} not in expected colors {expected_colors}"
            )

    def test_get_colors_for_plot_with_custom_multi_color_generator(self, fig_ax, styler):
        """Test get_colors_for_plot with custom multi-color generator (covers line 115)."""
        mock_single_gen = Mock()
        mock_multi_gen = Mock()
        mock_single_gen.__next__ = Mock(side_effect=["#FF0000", "#00FF00", "#0000FF"])
        mock_multi_gen.__next__ = Mock(side_effect=["#FF0000", "#00FF00", "#0000FF"])

        styler.context.get_color_generators = Mock(
            return_value={
                "single": mock_single_gen,
                "multi": mock_multi_gen,
            },
        )
        styler.context._custom_color_generators = True
        expected_color_count = 3
        colors = styler.get_colors_for_plot(expected_color_count)

        assert colors == ["#FF0000", "#00FF00", "#0000FF"]

        assert mock_multi_gen.__next__.call_count == expected_color_count
        assert mock_single_gen.__next__.call_count == 0
