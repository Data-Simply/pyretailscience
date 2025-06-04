"""Central configuration for fonts, colors, and styling options with proper bundled font management."""

import contextlib
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


class StylingContext:
    """Central styling context that maintains bundled fonts as defaults but enables customization."""

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
        """Get matplotlib FontProperties with flexible font resolution."""
        if font_name in self._font_cache:
            return self._font_cache[font_name]

        font_props = None

        if font_name in self._builtin_fonts:
            with contextlib.suppress(OSError, RuntimeError):
                font_props = fm.FontProperties(fname=self._builtin_fonts[font_name])

        elif font_name != "default":
            with contextlib.suppress(OSError, RuntimeError):
                font_props = fm.FontProperties(family=font_name)

        if font_props is None:
            font_props = fm.FontProperties()

        self._font_cache[font_name] = font_props
        return font_props


# Global styling context instance
_styling_context = StylingContext()


def get_styling_context() -> StylingContext:
    """Get the global styling context instance."""
    return _styling_context


def update_styling_context(context: StylingContext) -> None:
    """Update the global styling context (used by enterprise plugins)."""
    global _styling_context  # noqa:PLW0603
    _styling_context = context
