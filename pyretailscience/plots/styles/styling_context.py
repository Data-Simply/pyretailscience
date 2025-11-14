"""Central configuration for fonts, colors, and styling options with proper bundled font management."""

import importlib.resources as pkg_resources
from dataclasses import dataclass

import matplotlib.font_manager as fm

ASSETS_PATH = pkg_resources.files("pyretailscience").joinpath("assets")


@dataclass
class FontConfig:
    """Configuration for fonts used in plots."""

    title_font: str = "poppins_semi_bold"
    title_size: float = 20
    label_font: str = "poppins_regular"
    label_size: float = 12
    tick_font: str = "poppins_regular"
    tick_size: float = 10
    source_font: str = "poppins_light_italic"
    source_size: float = 10
    data_label_font: str = "poppins_regular"
    data_label_size: float = 8


class StylingContext:
    """Central styling context for fonts and styling options."""

    def __init__(self) -> None:
        """Initialize the styling context with default fonts and cache."""
        self.fonts = FontConfig()
        self._font_cache: dict[str, fm.FontProperties] = {}

        # Built-in font registry - maps names to bundled font files
        self._builtin_fonts = {
            "poppins_bold": f"{ASSETS_PATH}/fonts/Poppins-Bold.ttf",
            "poppins_semi_bold": f"{ASSETS_PATH}/fonts/Poppins-SemiBold.ttf",
            "poppins_regular": f"{ASSETS_PATH}/fonts/Poppins-Regular.ttf",
            "poppins_medium": f"{ASSETS_PATH}/fonts/Poppins-Medium.ttf",
            "poppins_light_italic": f"{ASSETS_PATH}/fonts/Poppins-LightItalic.ttf",
        }

    def get_font_properties(self, font_name: str) -> fm.FontProperties:
        """Get matplotlib FontProperties with flexible font resolution.

        This method resolves fonts in the following priority order:
        1. Check cache for previously loaded font
        2. Try to load from bundled fonts if font_name matches a built-in font
        3. Try to load as a system font family name

        Args:
            font_name (str): The name of the font to load. Can be:
                - A built-in font name (e.g., 'poppins_regular')
                - A system font family name (e.g., 'Arial', 'Times New Roman')

        Returns:
            fm.FontProperties: A matplotlib FontProperties object that can be used
                for text rendering.
        """
        if font_name in self._font_cache:
            return self._font_cache[font_name]

        # Try to load built-in bundled fonts first
        if font_name in self._builtin_fonts:
            font_props = fm.FontProperties(fname=self._builtin_fonts[font_name])
            self._font_cache[font_name] = font_props
            return font_props

        # Try to load as system font family
        font_props = fm.FontProperties(family=font_name)
        self._font_cache[font_name] = font_props
        return font_props


# Global styling context instance
_styling_context = StylingContext()


def get_styling_context() -> StylingContext:
    """Get the global styling context instance."""
    return _styling_context
