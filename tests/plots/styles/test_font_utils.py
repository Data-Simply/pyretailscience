"""Tests for font_utils module."""

import matplotlib.font_manager as fm
import pytest

from pyretailscience.options import get_option, option_context, set_option
from pyretailscience.plots.styles.font_utils import (
    get_font_config,
    get_font_properties,
    get_spacing_config,
    get_style_config,
)

DEFAULT_TITLE_SIZE = 20.0
DEFAULT_LABEL_SIZE = 12.0
DEFAULT_TICK_SIZE = 10.0
DEFAULT_SOURCE_SIZE = 10.0
DEFAULT_DATA_LABEL_SIZE = 8.0
DEFAULT_SPACING_PAD = 10
DEFAULT_GRID_ALPHA = 0.5
TEST_TITLE_SIZE = 24.0
TEST_SPACING_TITLE_PAD = 20
TEST_SPACING_X_LABEL_PAD = 15
TEST_GRID_ALPHA = 0.8
TEST_TITLE_SIZE_LARGE = 28.0
TEST_LABEL_SIZE = 14.0
TEST_FONT_SIZE_SMALL = 13.0
TEST_FONT_SIZE_MEDIUM = 22.0
LIGHT_COLOR_THRESHOLD = 0.9


class TestFontProperties:
    """Tests for get_font_properties function."""

    def test_get_bundled_font(self):
        """Test loading bundled Poppins fonts."""
        font_props = get_font_properties("poppins_regular")
        assert isinstance(font_props, fm.FontProperties)

        # Test different bundled fonts
        for font_name in ["poppins_bold", "poppins_semi_bold", "poppins_medium", "poppins_light_italic"]:
            font_props = get_font_properties(font_name)
            assert isinstance(font_props, fm.FontProperties)

    def test_get_system_font(self):
        """Test loading system fonts."""
        # Test common system fonts that should exist on most systems
        for font_name in ["Arial", "Helvetica", "serif", "sans-serif"]:
            try:
                font_props = get_font_properties(font_name)
                assert isinstance(font_props, fm.FontProperties)
            except ValueError:
                # Some fonts might not exist on the test system.
                pass

    def test_font_caching(self):
        """Test that fonts are cached for performance."""
        # First call
        font_props1 = get_font_properties("poppins_regular")
        # Second call should return the same cached object
        font_props2 = get_font_properties("poppins_regular")
        assert font_props1 is font_props2

    def test_invalid_font_raises_error(self):
        """Test that invalid font names raise ValueError."""
        with pytest.raises(ValueError, match="Font 'nonexistent_font' not found"):
            get_font_properties("nonexistent_font")


class TestConfigFunctions:
    """Tests for configuration getter functions."""

    def test_get_font_config_default_values(self):
        """Test font config returns expected default values."""
        config = get_font_config()

        assert config["title_font"] == "poppins_semi_bold"
        assert config["label_font"] == "poppins_regular"
        assert config["tick_font"] == "poppins_regular"
        assert config["source_font"] == "poppins_light_italic"
        assert config["data_label_font"] == "poppins_regular"
        assert config["title_size"] == DEFAULT_TITLE_SIZE
        assert config["label_size"] == DEFAULT_LABEL_SIZE
        assert config["tick_size"] == DEFAULT_TICK_SIZE
        assert config["source_size"] == DEFAULT_SOURCE_SIZE
        assert config["data_label_size"] == DEFAULT_DATA_LABEL_SIZE

    def test_get_spacing_config_default_values(self):
        """Test spacing config returns expected default values."""
        config = get_spacing_config()

        assert config["title_pad"] == DEFAULT_SPACING_PAD
        assert config["x_label_pad"] == DEFAULT_SPACING_PAD
        assert config["y_label_pad"] == DEFAULT_SPACING_PAD

    def test_get_style_config_default_values(self):
        """Test style config returns expected default values."""
        config = get_style_config()

        assert config["background_color"] == "white"
        assert config["grid_color"] == "#DAD8D7"
        assert config["grid_alpha"] == DEFAULT_GRID_ALPHA
        assert config["show_top_spine"] is False
        assert config["show_right_spine"] is False
        assert config["show_bottom_spine"] is True
        assert config["show_left_spine"] is True
        assert config["legend_bbox_to_anchor"] == [1.05, 1.0]
        assert config["legend_loc"] == "upper left"

    def test_font_config_respects_options(self):
        """Test that font config respects option changes."""
        with option_context("plot.font.title_font", "Arial", "plot.font.title_size", TEST_TITLE_SIZE):
            config = get_font_config()
            assert config["title_font"] == "Arial"
            assert config["title_size"] == TEST_TITLE_SIZE

        # Values should revert after context
        config = get_font_config()
        assert config["title_font"] == "poppins_semi_bold"
        assert config["title_size"] == DEFAULT_TITLE_SIZE

    def test_spacing_config_respects_options(self):
        """Test that spacing config respects option changes."""
        with option_context(
            "plot.spacing.title_pad",
            TEST_SPACING_TITLE_PAD,
            "plot.spacing.x_label_pad",
            TEST_SPACING_X_LABEL_PAD,
        ):
            config = get_spacing_config()
            assert config["title_pad"] == TEST_SPACING_TITLE_PAD
            assert config["x_label_pad"] == TEST_SPACING_X_LABEL_PAD

        # Values should revert after context
        config = get_spacing_config()
        assert config["title_pad"] == DEFAULT_SPACING_PAD
        assert config["x_label_pad"] == DEFAULT_SPACING_PAD

    def test_style_config_respects_options(self):
        """Test that style config respects option changes."""
        with option_context(
            "plot.style.background_color",
            "lightgray",
            "plot.style.grid_alpha",
            TEST_GRID_ALPHA,
            "plot.style.show_top_spine",
            True,
        ):
            config = get_style_config()
            assert config["background_color"] == "lightgray"
            assert config["grid_alpha"] == TEST_GRID_ALPHA
            assert config["show_top_spine"] is True

        # Values should revert after context
        config = get_style_config()
        assert config["background_color"] == "white"
        assert config["grid_alpha"] == DEFAULT_GRID_ALPHA
        assert config["show_top_spine"] is False


class TestOptionsIntegration:
    """Tests for integration with the options system."""

    def test_all_font_options_exist(self):
        """Test that all font-related options are properly registered."""
        font_options = [
            "plot.font.title_font",
            "plot.font.label_font",
            "plot.font.tick_font",
            "plot.font.source_font",
            "plot.font.data_label_font",
            "plot.font.title_size",
            "plot.font.label_size",
            "plot.font.tick_size",
            "plot.font.source_size",
            "plot.font.data_label_size",
        ]

        for option in font_options:
            # Should not raise ValueError
            value = get_option(option)
            assert value is not None

    def test_all_spacing_options_exist(self):
        """Test that all spacing-related options are properly registered."""
        spacing_options = [
            "plot.spacing.title_pad",
            "plot.spacing.x_label_pad",
            "plot.spacing.y_label_pad",
        ]

        for option in spacing_options:
            # Should not raise ValueError
            value = get_option(option)
            assert value is not None

    def test_all_style_options_exist(self):
        """Test that all style-related options are properly registered."""
        style_options = [
            "plot.style.background_color",
            "plot.style.grid_color",
            "plot.style.grid_alpha",
            "plot.style.show_top_spine",
            "plot.style.show_right_spine",
            "plot.style.show_bottom_spine",
            "plot.style.show_left_spine",
            "plot.style.legend_bbox_to_anchor",
            "plot.style.legend_loc",
        ]

        for option in style_options:
            # Should not raise ValueError
            value = get_option(option)
            assert value is not None

    def test_font_option_modifications(self):
        """Test that font options can be modified and affect config."""
        original_title_font = get_option("plot.font.title_font")
        original_title_size = get_option("plot.font.title_size")

        try:
            # Modify options
            set_option("plot.font.title_font", "Arial")
            set_option("plot.font.title_size", TEST_TITLE_SIZE_LARGE)

            # Check that config reflects changes
            config = get_font_config()
            assert config["title_font"] == "Arial"
            assert config["title_size"] == TEST_TITLE_SIZE_LARGE

        finally:
            # Reset to original values
            set_option("plot.font.title_font", original_title_font)
            set_option("plot.font.title_size", original_title_size)
