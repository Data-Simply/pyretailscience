"""Integration tests for styling options with actual plots."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pyretailscience.options import option_context
from pyretailscience.plots import scatter

TEST_TITLE_SIZE = 24.0
TEST_LABEL_SIZE = 14.0
TEST_FONT_SIZE_SMALL = 13.0
TEST_FONT_SIZE_MEDIUM = 22.0
LIGHT_COLOR_THRESHOLD = 0.9


class TestStylingOptionsIntegration:
    """Tests for integration of styling options with actual plot functions."""

    @pytest.fixture(autouse=True)
    def cleanup_figures(self):
        """Clean up matplotlib figures after each test."""
        yield
        plt.close("all")

    @pytest.fixture
    def sample_dataframe(self):
        """Sample data for testing plots."""
        return pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 5, 3, 8, 7],
                "category": ["A", "A", "B", "B", "A"],
                "labels": ["Point1", "Point2", "Point3", "Point4", "Point5"],
            },
        )

    def test_font_customization_with_context(self, sample_dataframe):
        """Test that font options affect plot appearance via context manager."""
        # Test with custom font settings using available bundled fonts
        with option_context(
            "plot.font.title_font",
            "poppins_bold",
            "plot.font.title_size",
            24.0,
            "plot.font.label_size",
            14.0,
        ):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                title="Custom Font Test",
                x_label="X Axis",
                y_label="Y Axis",
            )

            # Verify title properties
            assert ax.get_title() == "Custom Font Test"

            assert ax.title.get_fontsize() == TEST_TITLE_SIZE

            # Verify label properties
            assert ax.get_xlabel() == "X Axis"
            assert ax.get_ylabel() == "Y Axis"
            assert ax.xaxis.label.get_fontsize() == TEST_LABEL_SIZE
            assert ax.yaxis.label.get_fontsize() == TEST_LABEL_SIZE

    def test_spacing_customization_with_context(self, sample_dataframe):
        """Test that spacing options affect plot layout via context manager."""
        with option_context(
            "plot.spacing.title_pad",
            25,
            "plot.spacing.x_label_pad",
            20,
            "plot.spacing.y_label_pad",
            15,
        ):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                title="Custom Spacing Test",
                x_label="X Axis",
                y_label="Y Axis",
            )

            assert ax.get_title() == "Custom Spacing Test"
            assert ax.get_xlabel() == "X Axis"
            assert ax.get_ylabel() == "Y Axis"

    def test_style_customization_with_context(self, sample_dataframe):
        """Test that style options affect plot appearance via context manager."""
        with option_context(
            "plot.style.background_color",
            "lightgray",
            "plot.style.grid_alpha",
            0.8,
            "plot.style.show_top_spine",
            True,
            "plot.style.show_right_spine",
            True,
        ):
            ax = scatter.plot(df=sample_dataframe, value_col="y", x_col="x", title="Custom Style Test")

            # Verify background color
            assert ax.get_facecolor() == (
                0.8274509803921568,
                0.8274509803921568,
                0.8274509803921568,
                1.0,
            )  # lightgray in RGBA

            # Verify spine visibility
            assert ax.spines["top"].get_visible() is True
            assert ax.spines["right"].get_visible() is True
            assert ax.spines["bottom"].get_visible() is True
            assert ax.spines["left"].get_visible() is True

    def test_label_font_customization(self, sample_dataframe):
        """Test that data label fonts can be customized."""
        with option_context("plot.font.data_label_size", 12.0):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                label_col="labels",
                title="Custom Label Font Test",
            )

            assert ax.get_title() == "Custom Label Font Test"

    def test_multiple_option_changes(self, sample_dataframe):
        """Test that multiple styling options work together."""
        with option_context(
            # Font options
            "plot.font.title_font",
            "poppins_medium",
            "plot.font.title_size",
            TEST_FONT_SIZE_MEDIUM,
            "plot.font.label_size",
            TEST_FONT_SIZE_SMALL,
            # Spacing options
            "plot.spacing.title_pad",
            20,
            # Style options
            "plot.style.grid_alpha",
            0.3,
            "plot.style.show_top_spine",
            True,
            "plot.style.background_color",
            "whitesmoke",
        ):
            ax = scatter.plot(
                df=sample_dataframe,
                value_col="y",
                x_col="x",
                group_col="category",
                title="Multi-Option Test",
                x_label="X Values",
                y_label="Y Values",
            )

            # Verify various properties
            assert ax.get_title() == "Multi-Option Test"
            assert ax.title.get_fontsize() == TEST_FONT_SIZE_MEDIUM
            assert ax.get_xlabel() == "X Values"
            assert ax.get_ylabel() == "Y Values"
            assert ax.xaxis.label.get_fontsize() == TEST_FONT_SIZE_SMALL
            assert ax.yaxis.label.get_fontsize() == TEST_FONT_SIZE_SMALL

            # Verify style changes
            assert ax.spines["top"].get_visible() is True
            # Background color is approximately whitesmoke
            bg_color = ax.get_facecolor()
            assert all(c > LIGHT_COLOR_THRESHOLD for c in bg_color[:3])  # Very light color

    def test_default_behavior_unchanged(self, sample_dataframe):
        """Test that default behavior works when no options are changed."""
        ax = scatter.plot(
            df=sample_dataframe,
            value_col="y",
            x_col="x",
            title="Default Behavior Test",
            x_label="X Axis",
            y_label="Y Axis",
        )

        # Default behavior should still work
        assert ax.get_title() == "Default Behavior Test"
        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"

        # Default spines configuration
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["bottom"].get_visible() is True
        assert ax.spines["left"].get_visible() is True

        # Default background should be white
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white in RGBA
