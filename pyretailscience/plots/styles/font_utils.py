"""Font utilities for plot styling with options system integration."""

import importlib.resources as pkg_resources

import matplotlib.font_manager as fm

from pyretailscience.options import get_option

ASSETS_PATH = pkg_resources.files("pyretailscience").joinpath("assets")

# Built-in font registry - maps names to bundled font files
_BUILTIN_FONTS = {
    "poppins_bold": f"{ASSETS_PATH}/fonts/Poppins-Bold.ttf",
    "poppins_semi_bold": f"{ASSETS_PATH}/fonts/Poppins-SemiBold.ttf",
    "poppins_regular": f"{ASSETS_PATH}/fonts/Poppins-Regular.ttf",
    "poppins_medium": f"{ASSETS_PATH}/fonts/Poppins-Medium.ttf",
    "poppins_light_italic": f"{ASSETS_PATH}/fonts/Poppins-LightItalic.ttf",
}

# Font cache for performance
_font_cache: dict[str, fm.FontProperties] = {}


def _raise_font_not_found_error(font_name: str) -> None:
    """Raise a ValueError for font not found.

    Args:
        font_name (str): The name of the font that was not found.

    Raises:
        ValueError: Always raises with a descriptive error message.
    """
    error_msg = f"Font '{font_name}' not found in system fonts"
    raise ValueError(error_msg)


def get_font_properties(font_name: str) -> fm.FontProperties:
    """Get matplotlib FontProperties with flexible font resolution.

    This function resolves fonts in the following priority order:
    1. Check cache for previously loaded font
    2. Try to load from bundled fonts if font_name matches a built-in font
    3. Try to load as a system font family name
    4. Raise ValueError if font cannot be found (strict validation)

    Args:
        font_name (str): The name of the font to load. Can be:
            - A built-in font name (e.g., 'poppins_regular')
            - A system font family name (e.g., 'Arial', 'Times New Roman')

    Returns:
        fm.FontProperties: A matplotlib FontProperties object that can be used
            for text rendering.

    Raises:
        ValueError: If the font cannot be found in bundled fonts or system fonts.
    """
    if font_name in _font_cache:
        return _font_cache[font_name]

    # Try to load built-in bundled fonts first
    if font_name in _BUILTIN_FONTS:
        font_props = fm.FontProperties(fname=_BUILTIN_FONTS[font_name])
        _font_cache[font_name] = font_props
        return font_props

    # Try to load as system font family - validate it exists
    try:
        # Create a FontProperties object and verify the font exists
        font_props = fm.FontProperties(family=font_name)

        # Use matplotlib's font finder to validate the font exists
        font_finder = fm.fontManager.findfont(font_props, fallback_to_default=False)

        # If findfont returns the default font path, the requested font wasn't found
        default_font_path = fm.fontManager.defaultFont["ttf"]
        if font_finder == default_font_path:
            _raise_font_not_found_error(font_name)
        else:
            _font_cache[font_name] = font_props
            return font_props
    except Exception as e:
        error_msg = f"Font '{font_name}' not found. Available bundled fonts: {list(_BUILTIN_FONTS.keys())}"
        raise ValueError(error_msg) from e


def get_font_config() -> dict[str, any]:
    """Get current font configuration from options.

    Returns:
        dict: Current font configuration with font names and sizes.
    """
    return {
        "title_font": get_option("plot.font.title_font"),
        "label_font": get_option("plot.font.label_font"),
        "tick_font": get_option("plot.font.tick_font"),
        "source_font": get_option("plot.font.source_font"),
        "data_label_font": get_option("plot.font.data_label_font"),
        "title_size": get_option("plot.font.title_size"),
        "label_size": get_option("plot.font.label_size"),
        "tick_size": get_option("plot.font.tick_size"),
        "source_size": get_option("plot.font.source_size"),
        "data_label_size": get_option("plot.font.data_label_size"),
    }


def get_spacing_config() -> dict[str, any]:
    """Get current spacing configuration from options.

    Returns:
        dict: Current spacing configuration with padding values.
    """
    return {
        "title_pad": get_option("plot.spacing.title_pad"),
        "x_label_pad": get_option("plot.spacing.x_label_pad"),
        "y_label_pad": get_option("plot.spacing.y_label_pad"),
    }


def get_style_config() -> dict[str, any]:
    """Get current style configuration from options.

    Returns:
        dict: Current style configuration with colors, grid settings, etc.
    """
    return {
        "background_color": get_option("plot.style.background_color"),
        "grid_color": get_option("plot.style.grid_color"),
        "grid_alpha": get_option("plot.style.grid_alpha"),
        "show_top_spine": get_option("plot.style.show_top_spine"),
        "show_right_spine": get_option("plot.style.show_right_spine"),
        "show_bottom_spine": get_option("plot.style.show_bottom_spine"),
        "show_left_spine": get_option("plot.style.show_left_spine"),
        "legend_bbox_to_anchor": get_option("plot.style.legend_bbox_to_anchor"),
        "legend_loc": get_option("plot.style.legend_loc"),
    }
