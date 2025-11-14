"""Test for the StylingContext configuration."""

import matplotlib.font_manager as fm
import pytest

from pyretailscience.plots.styles.styling_context import (
    FontConfig,
    StylingContext,
)


class TestStylingContext:
    """Unit tests for StylingContext class."""

    @pytest.fixture
    def context(self):
        """Fixture to provide a fresh StylingContext instance."""
        return StylingContext()

    @pytest.mark.parametrize(
        "font_name",
        [
            "poppins_bold",
            "poppins_semi_bold",
            "poppins_regular",
            "poppins_medium",
            "poppins_light_italic",
        ],
    )
    def test_font_resolution_builtin_fonts(self, context, font_name):
        """Test bundled font names resolve correctly."""
        props = context.get_font_properties(font_name)
        assert props is not None
        assert isinstance(props, fm.FontProperties)

    @pytest.mark.parametrize(
        "font_name",
        [
            "/nonexistent/font.ttf",  # Custom path that doesn't exist
            "NonExistentFontFamily",  # Family name that doesn't exist
        ],
    )
    def test_custom_font_fallback(self, context, font_name):
        """Test graceful fallback when custom fonts are unavailable."""
        props = context.get_font_properties(font_name)
        assert props is not None
        assert isinstance(props, fm.FontProperties)

    def test_default_font_fallback(self, context):
        """Test fallback to default font."""
        props = context.get_font_properties("default")
        assert props is not None
        assert isinstance(props, fm.FontProperties)

    def test_font_caching(self, context):
        """Test that fonts are cached after first retrieval."""
        # First call
        props1 = context.get_font_properties("poppins_regular")

        # Second call should return cached version
        props2 = context.get_font_properties("poppins_regular")

        assert props1 is props2
        assert "poppins_regular" in context._font_cache

    def test_font_config_customization(self):
        """Test FontConfig can be customized."""
        custom_title_size = 24
        custom_label_size = 14
        expected_tick_size = 10
        config = FontConfig(
            title_font="custom_font",
            title_size=custom_title_size,
            label_size=custom_label_size,
        )

        assert config.title_font == "custom_font"
        assert config.title_size == custom_title_size
        assert config.label_size == custom_label_size
        assert config.label_font == "poppins_regular"
        assert config.tick_size == expected_tick_size
