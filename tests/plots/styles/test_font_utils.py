"""Tests for font_utils module."""

import matplotlib.font_manager as fm
import pytest

from pyretailscience.plots.styles.font_utils import get_font_properties


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
        """Test loading generic font families that matplotlib always supports."""
        for font_name in ["serif", "sans", "monospace", "DejaVu Sans"]:
            font_props = get_font_properties(font_name)
            assert isinstance(font_props, fm.FontProperties)

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
