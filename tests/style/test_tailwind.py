"""Tests for the tailwind module in the style package."""

from itertools import islice

from pyretailscience.style.tailwind import COLORS, get_multi_color_cmap, get_single_color_cmap


def test_get_single_color_cmap_three_colors():
    """Test the get_single_color_cmap function with three shades of green."""
    gen = get_single_color_cmap()
    expected_colors = [
        COLORS["green"][500],
        COLORS["green"][300],
        COLORS["green"][700],
    ]

    # Get the first three colors from the generator
    generated_colors = list(islice(gen, 3))

    # Ensure the generated colors match the expected green shades
    assert generated_colors == expected_colors


def test_get_single_color_cmap_cycle_behavior():
    """Test the cycling behavior of the get_single_color_cmap function."""
    gen = get_single_color_cmap()

    # Get the first six colors (should cycle after the third)
    generated_colors = list(islice(gen, 6))
    expected_colors = [
        COLORS["green"][500],
        COLORS["green"][300],
        COLORS["green"][700],
        COLORS["green"][500],
        COLORS["green"][300],
        COLORS["green"][700],
    ]

    # Ensure the generated colors match the expected cycling behavior
    assert generated_colors == expected_colors


def test_get_multi_color_cmap_seven_colors():
    """Test the get_multi_color_cmap function with seven colors."""
    gen = get_multi_color_cmap()
    expected_colors = [
        COLORS["green"][500],
        COLORS["blue"][500],
        COLORS["red"][500],
        COLORS["orange"][500],
        COLORS["yellow"][500],
        COLORS["violet"][500],
        COLORS["pink"][500],
    ]

    # Get the first seven colors from the generator
    generated_colors = list(islice(gen, 7))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors


def test_get_multi_color_cmap_cycle_behavior():
    """Test the cycling behavior of the get_multi_color_cmap function."""
    gen = get_multi_color_cmap()

    # Get the first 14 colors (should cycle after the 7th)
    generated_colors = list(islice(gen, 14))
    expected_colors = [
        COLORS["green"][500],
        COLORS["blue"][500],
        COLORS["red"][500],
        COLORS["orange"][500],
        COLORS["yellow"][500],
        COLORS["violet"][500],
        COLORS["pink"][500],
        COLORS["green"][300],
        COLORS["blue"][300],
        COLORS["red"][300],
        COLORS["orange"][300],
        COLORS["yellow"][300],
        COLORS["violet"][300],
        COLORS["pink"][300],
    ]

    # Ensure the generated colors match the expected cycling behavior
    assert generated_colors == expected_colors


def test_get_single_color_cmap_two_colors():
    """Test the get_single_color_cmap function with two shades of green."""
    gen = get_single_color_cmap()
    expected_colors = [
        COLORS["green"][500],
        COLORS["green"][300],
    ]

    # Get the first two colors from the generator
    generated_colors = list(islice(gen, 2))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors


def test_get_multi_color_cmap_five_colors():
    """Test the get_multi_color_cmap function with five colors."""
    gen = get_multi_color_cmap()
    expected_colors = [
        COLORS["green"][500],
        COLORS["blue"][500],
        COLORS["red"][500],
        COLORS["orange"][500],
        COLORS["yellow"][500],
    ]

    # Get the first five colors from the generator
    generated_colors = list(islice(gen, 5))

    # Ensure the generated colors match the expected ones
    assert generated_colors == expected_colors
