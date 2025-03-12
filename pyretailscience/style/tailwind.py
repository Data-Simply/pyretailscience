"""Tailwind CSS color Palettes and helper functions.

PyRetailScience includes the raw Tailwind CSS color palettes and ListedColormaps and LinearSegmentedColormaps versions
for use when charting.

Colors from Tailwind CSS
https://raw.githubusercontent.com/tailwindlabs/tailwindcss/a1e74f055b13a7ef5775bdd72a77a4d397565016/src/public/colors.js

"""

from collections.abc import Generator
from itertools import cycle

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

COLORS = {
    "slate": {
        50: "#f8fafc",
        100: "#f1f5f9",
        200: "#e2e8f0",
        300: "#cbd5e1",
        400: "#94a3b8",
        500: "#64748b",
        600: "#475569",
        700: "#334155",
        800: "#1e293b",
        900: "#0f172a",
        950: "#020617",
    },
    "gray": {
        50: "#f9fafb",
        100: "#f3f4f6",
        200: "#e5e7eb",
        300: "#d1d5db",
        400: "#9ca3af",
        500: "#6b7280",
        600: "#4b5563",
        700: "#374151",
        800: "#1f2937",
        900: "#111827",
        950: "#030712",
    },
    "zinc": {
        50: "#fafafa",
        100: "#f4f4f5",
        200: "#e4e4e7",
        300: "#d4d4d8",
        400: "#a1a1aa",
        500: "#71717a",
        600: "#52525b",
        700: "#3f3f46",
        800: "#27272a",
        900: "#18181b",
        950: "#09090b",
    },
    "neutral": {
        50: "#fafafa",
        100: "#f5f5f5",
        200: "#e5e5e5",
        300: "#d4d4d4",
        400: "#a3a3a3",
        500: "#737373",
        600: "#525252",
        700: "#404040",
        800: "#262626",
        900: "#171717",
        950: "#0a0a0a",
    },
    "stone": {
        50: "#fafaf9",
        100: "#f5f5f4",
        200: "#e7e5e4",
        300: "#d6d3d1",
        400: "#a8a29e",
        500: "#78716c",
        600: "#57534e",
        700: "#44403c",
        800: "#292524",
        900: "#1c1917",
        950: "#0c0a09",
    },
    "red": {
        50: "#fef2f2",
        100: "#fee2e2",
        200: "#fecaca",
        300: "#fca5a5",
        400: "#f87171",
        500: "#ef4444",
        600: "#dc2626",
        700: "#b91c1c",
        800: "#991b1b",
        900: "#7f1d1d",
        950: "#450a0a",
    },
    "orange": {
        50: "#fff7ed",
        100: "#ffedd5",
        200: "#fed7aa",
        300: "#fdba74",
        400: "#fb923c",
        500: "#f97316",
        600: "#ea580c",
        700: "#c2410c",
        800: "#9a3412",
        900: "#7c2d12",
        950: "#431407",
    },
    "amber": {
        50: "#fffbeb",
        100: "#fef3c7",
        200: "#fde68a",
        300: "#fcd34d",
        400: "#fbbf24",
        500: "#f59e0b",
        600: "#d97706",
        700: "#b45309",
        800: "#92400e",
        900: "#78350f",
        950: "#451a03",
    },
    "yellow": {
        50: "#fefce8",
        100: "#fef9c3",
        200: "#fef08a",
        300: "#fde047",
        400: "#facc15",
        500: "#eab308",
        600: "#ca8a04",
        700: "#a16207",
        800: "#854d0e",
        900: "#713f12",
        950: "#422006",
    },
    "lime": {
        50: "#f7fee7",
        100: "#ecfccb",
        200: "#d9f99d",
        300: "#bef264",
        400: "#a3e635",
        500: "#84cc16",
        600: "#65a30d",
        700: "#4d7c0f",
        800: "#3f6212",
        900: "#365314",
        950: "#1a2e05",
    },
    "green": {
        50: "#f0fdf4",
        100: "#dcfce7",
        200: "#bbf7d0",
        300: "#86efac",
        400: "#4ade80",
        500: "#22c55e",
        600: "#16a34a",
        700: "#15803d",
        800: "#166534",
        900: "#14532d",
        950: "#052e16",
    },
    "emerald": {
        50: "#ecfdf5",
        100: "#d1fae5",
        200: "#a7f3d0",
        300: "#6ee7b7",
        400: "#34d399",
        500: "#10b981",
        600: "#059669",
        700: "#047857",
        800: "#065f46",
        900: "#064e3b",
        950: "#022c22",
    },
    "teal": {
        50: "#f0fdfa",
        100: "#ccfbf1",
        200: "#99f6e4",
        300: "#5eead4",
        400: "#2dd4bf",
        500: "#14b8a6",
        600: "#0d9488",
        700: "#0f766e",
        800: "#115e59",
        900: "#134e4a",
        950: "#042f2e",
    },
    "cyan": {
        50: "#ecfeff",
        100: "#cffafe",
        200: "#a5f3fc",
        300: "#67e8f9",
        400: "#22d3ee",
        500: "#06b6d4",
        600: "#0891b2",
        700: "#0e7490",
        800: "#155e75",
        900: "#164e63",
        950: "#083344",
    },
    "sky": {
        50: "#f0f9ff",
        100: "#e0f2fe",
        200: "#bae6fd",
        300: "#7dd3fc",
        400: "#38bdf8",
        500: "#0ea5e9",
        600: "#0284c7",
        700: "#0369a1",
        800: "#075985",
        900: "#0c4a6e",
        950: "#082f49",
    },
    "blue": {
        50: "#eff6ff",
        100: "#dbeafe",
        200: "#bfdbfe",
        300: "#93c5fd",
        400: "#60a5fa",
        500: "#3b82f6",
        600: "#2563eb",
        700: "#1d4ed8",
        800: "#1e40af",
        900: "#1e3a8a",
        950: "#172554",
    },
    "indigo": {
        50: "#eef2ff",
        100: "#e0e7ff",
        200: "#c7d2fe",
        300: "#a5b4fc",
        400: "#818cf8",
        500: "#6366f1",
        600: "#4f46e5",
        700: "#4338ca",
        800: "#3730a3",
        900: "#312e81",
        950: "#1e1b4b",
    },
    "violet": {
        50: "#f5f3ff",
        100: "#ede9fe",
        200: "#ddd6fe",
        300: "#c4b5fd",
        400: "#a78bfa",
        500: "#8b5cf6",
        600: "#7c3aed",
        700: "#6d28d9",
        800: "#5b21b6",
        900: "#4c1d95",
        950: "#2e1065",
    },
    "purple": {
        50: "#faf5ff",
        100: "#f3e8ff",
        200: "#e9d5ff",
        300: "#d8b4fe",
        400: "#c084fc",
        500: "#a855f7",
        600: "#9333ea",
        700: "#7e22ce",
        800: "#6b21a8",
        900: "#581c87",
        950: "#3b0764",
    },
    "fuchsia": {
        50: "#fdf4ff",
        100: "#fae8ff",
        200: "#f5d0fe",
        300: "#f0abfc",
        400: "#e879f9",
        500: "#d946ef",
        600: "#c026d3",
        700: "#a21caf",
        800: "#86198f",
        900: "#701a75",
        950: "#4a044e",
    },
    "pink": {
        50: "#fdf2f8",
        100: "#fce7f3",
        200: "#fbcfe8",
        300: "#f9a8d4",
        400: "#f472b6",
        500: "#ec4899",
        600: "#db2777",
        700: "#be185d",
        800: "#9d174d",
        900: "#831843",
        950: "#500724",
    },
    "rose": {
        50: "#fff1f2",
        100: "#ffe4e6",
        200: "#fecdd3",
        300: "#fda4af",
        400: "#fb7185",
        500: "#f43f5e",
        600: "#e11d48",
        700: "#be123c",
        800: "#9f1239",
        900: "#881337",
        950: "#4c0519",
    },
}


def get_color_list(name: str, starting_color_code: int = 50, ending_color_code: int = 950) -> list[str]:
    """Returns a filtered list of colors from the Tailwind color palette based on the given range.

    Args:
        name (str): The name of the color palette (e.g., "blue", "red").
        starting_color_code (int): The lowest color shade to use (default: 50).
        ending_color_code (int): The highest color shade to use (default: 950).

    Returns:
        list[str]: A filtered list of colors from the Tailwind color palette.
    """
    if name not in COLORS:
        msg = f"Color pallete {name} not found. Available color palettes are: {', '.join(COLORS.keys())}."
        raise ValueError(msg)
    return [COLORS[name][key] for key in sorted(COLORS[name].keys()) if starting_color_code <= key <= ending_color_code]


def get_listed_cmap(name: str) -> ListedColormap:
    """Returns a ListedColormap from the Tailwind color pallete of the given name.

    Args:
        name (str): The name of the color pallete.

    Returns:
        ListedColormap: The color pallete as a ListedColormap.
    """
    return ListedColormap(get_color_list(name))


def get_linear_cmap(name: str, starting_color_code: int = 50, ending_color_code: int = 950) -> LinearSegmentedColormap:
    """Returns a linear segmented colormap using Tailwind colors.

    This function allows restricting the color range used in the colormap.

    Args:
        name (str): The name of the Tailwind color (e.g., "blue", "red").
        starting_color_code (int): The lowest color shade to use (default: 50).
        ending_color_code (int): The highest color shade to use (default: 950).

    Returns:
        LinearSegmentedColormap: A colormap object for matplotlib.
    """
    return LinearSegmentedColormap.from_list(
        f"{name}_linear_colormap",
        get_color_list(name, starting_color_code, ending_color_code),
    )


def get_base_cmap() -> ListedColormap:
    """Returns a ListedColormap with all the Tailwind colors.

    Returns:
        ListedColormap: A ListedColormap with all the Tailwind colors.
    """
    color_order = [
        "red",
        "orange",
        "yellow",
        "green",
        "teal",
        "sky",
        "indigo",
        "purple",
        "pink",
        "slate",
        "amber",
        "lime",
        "emerald",
        "cyan",
        "blue",
        "violet",
        "fuchsia",
        "rose",
    ]
    color_numbers = [500, 300, 700]
    colors = [COLORS[color][color_number] for color_number in color_numbers for color in color_order]

    return ListedColormap(colors)


def get_single_color_cmap() -> Generator[str, None, None]:
    """Returns a generator for Tailwind green shades.

    Returns:
        Generator: A generator yielding green shades in a looping fashion.
    """
    color_numbers = [500, 300, 700]
    return cycle([COLORS["green"][shade] for shade in color_numbers])


def get_multi_color_cmap() -> Generator[str, None, None]:
    """Returns a generator for multiple Tailwind colors and shades.

    Returns:
        Generator: A generator yielding the required colors in a looping fashion.
    """
    color_order = ["green", "blue", "red", "orange", "yellow", "violet", "pink"]
    color_numbers = [500, 300, 700]

    return cycle([COLORS[color][color_number] for color_number in color_numbers for color in color_order])


slate_cmap = get_listed_cmap("slate")
gray_cmap = get_listed_cmap("gray")
zinc_cmap = get_listed_cmap("zinc")
neutral_cmap = get_listed_cmap("neutral")
stone_cmap = get_listed_cmap("stone")
red_cmap = get_listed_cmap("red")
orange_cmap = get_listed_cmap("orange")
amber_cmap = get_listed_cmap("amber")
yellow_cmap = get_listed_cmap("yellow")
lime_cmap = get_listed_cmap("lime")
green_cmap = get_listed_cmap("green")
emerald_cmap = get_listed_cmap("emerald")
teal_cmap = get_listed_cmap("teal")
cyan_cmap = get_listed_cmap("cyan")
sky_cmap = get_listed_cmap("sky")
blue_cmap = get_listed_cmap("blue")
indigo_cmap = get_listed_cmap("indigo")
violet_cmap = get_listed_cmap("violet")
purple_cmap = get_listed_cmap("purple")
fuchsia_cmap = get_listed_cmap("fuchsia")
pink_cmap = get_listed_cmap("pink")


slate_linear_cmap = get_linear_cmap("slate")
gray_linear_cmap = get_linear_cmap("gray")
zinc_linear_cmap = get_linear_cmap("zinc")
neutral_linear_cmap = get_linear_cmap("neutral")
stone_linear_cmap = get_linear_cmap("stone")
red_linear_cmap = get_linear_cmap("red")
orange_linear_cmap = get_linear_cmap("orange")
amber_linear_cmap = get_linear_cmap("amber")
yellow_linear_cmap = get_linear_cmap("yellow")
lime_linear_cmap = get_linear_cmap("lime")
green_linear_cmap = get_linear_cmap("green")
emerald_linear_cmap = get_linear_cmap("emerald")
teal_linear_cmap = get_linear_cmap("teal")
cyan_linear_cmap = get_linear_cmap("cyan")
sky_linear_cmap = get_linear_cmap("sky")
blue_linear_cmap = get_linear_cmap("blue")
indigo_linear_cmap = get_linear_cmap("indigo")
violet_linear_cmap = get_linear_cmap("violet")
purple_linear_cmap = get_linear_cmap("purple")
fuchsia_linear_cmap = get_linear_cmap("fuchsia")
pink_linear_cmap = get_linear_cmap("pink")


base_cmap = get_base_cmap()
