"""Tests for the plots.tree_diagram module."""

import matplotlib.pyplot as plt
import pytest

from pyretailscience.plots.tree_diagram import BaseRoundedBox


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def ax():
    """Create a matplotlib axes for testing."""
    _, ax = plt.subplots()
    return ax


class TestBaseRoundedBox:
    """Test the BaseRoundedBox class."""

    # Path vertices counts (based on BaseRoundedBox implementation)
    ARC_POINTS_PER_CORNER = 10  # Number of points used to draw each rounded corner
    SQUARE_CORNER_VERTICES = 5  # 4 corners + 1 close for square box
    ROUNDED_CORNER_VERTICES = 4 * ARC_POINTS_PER_CORNER + 1  # 4 corners * 10 points + 1 close

    def test_box_creation_with_dimensions(self, ax):
        """Test that box is created with correct dimensions."""
        box = BaseRoundedBox(xy=(0, 0), width=3.0, height=2.0)
        ax.add_patch(box)

        # Verify the box was created
        assert box is not None
        # Verify it's a patch
        assert hasattr(box, "get_path")

    def test_box_positioning(self, ax):
        """Test that box is positioned at correct coordinates and bbox covers expected area."""
        x, y = 1.5, 2.5
        width, height = 3.0, 2.0
        box = BaseRoundedBox(xy=(x, y), width=width, height=height)
        ax.add_patch(box)

        # Get the bounding box
        bbox = box.get_path().get_extents()

        # The box should start at (x, y) and extend by width and height
        # Note: path extents may vary slightly due to rounded corners
        assert bbox.x0 == pytest.approx(x, abs=0.1)
        assert bbox.y0 == pytest.approx(y, abs=0.1)
        assert bbox.x1 == pytest.approx(x + width, abs=0.1)
        assert bbox.y1 == pytest.approx(y + height, abs=0.1)

    def test_linewidth_property(self, ax):
        """Test that linewidth is set correctly."""
        linewidth = 2.5
        box = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, linewidth=linewidth)
        ax.add_patch(box)

        assert box.get_linewidth() == linewidth

    def test_border_radius_top(self, ax):
        """Test that top_radius affects the path correctly."""
        box_with_radius = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.5, bottom_radius=0.0)
        box_no_radius = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.0, bottom_radius=0.0)

        # Boxes with different radii should have different paths
        assert len(box_with_radius.get_path().vertices) != len(box_no_radius.get_path().vertices)

    def test_border_radius_bottom(self, ax):
        """Test that bottom_radius affects the path correctly."""
        box_with_radius = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.0, bottom_radius=0.5)
        box_no_radius = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.0, bottom_radius=0.0)

        # Boxes with different radii should have different paths
        assert len(box_with_radius.get_path().vertices) != len(box_no_radius.get_path().vertices)

    def test_rendering_to_axes(self, ax):
        """Test that patch is added to axes.patches collection."""
        initial_patch_count = len(ax.patches)

        box = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0)
        ax.add_patch(box)

        # Verify the patch was added to the axes
        assert len(ax.patches) == initial_patch_count + 1
        assert box in ax.patches

    def test_zero_radius_creates_square_corners(self, ax):
        """Test that zero radius creates a box with square corners."""
        box = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.0, bottom_radius=0.0)
        ax.add_patch(box)

        # With zero radius, we should have fewer vertices (just the corners, not arc points)
        path = box.get_path()
        assert len(path.vertices) == self.SQUARE_CORNER_VERTICES

    def test_nonzero_radius_creates_rounded_corners(self, ax):
        """Test that nonzero radius creates more vertices for rounded corners."""
        box = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.3, bottom_radius=0.3)
        ax.add_patch(box)

        # With rounded corners, we should have many more vertices (10 per corner arc)
        path = box.get_path()
        assert len(path.vertices) == self.ROUNDED_CORNER_VERTICES

    def test_different_top_bottom_radius(self, ax):
        """Test that different top and bottom radii produce different shapes."""
        box_top_only = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.5, bottom_radius=0.0)
        box_bottom_only = BaseRoundedBox(xy=(0, 0), width=2.0, height=1.0, top_radius=0.0, bottom_radius=0.5)

        # Different radius configurations should produce different paths
        assert not (box_top_only.get_path().vertices == box_bottom_only.get_path().vertices).all()

    def test_multiple_boxes_on_same_axes(self, ax):
        """Test creating multiple boxes on the same axes."""
        box1 = BaseRoundedBox(xy=(0, 0), width=1.0, height=1.0, facecolor="red")
        box2 = BaseRoundedBox(xy=(2, 0), width=1.0, height=1.0, facecolor="blue")
        box3 = BaseRoundedBox(xy=(4, 0), width=1.0, height=1.0, facecolor="green")

        ax.add_patch(box1)
        ax.add_patch(box2)
        ax.add_patch(box3)

        # Verify all three boxes are in the axes
        expected_patch_count = 3
        assert len(ax.patches) == expected_patch_count
        assert box1 in ax.patches
        assert box2 in ax.patches
        assert box3 in ax.patches

    def test_kwargs_passed_to_patch(self, ax):
        """Test that additional kwargs are passed to the PathPatch."""
        expected_linewidth = 3
        expected_alpha = 0.5

        box = BaseRoundedBox(
            xy=(0, 0),
            width=2.0,
            height=1.0,
            facecolor="yellow",
            edgecolor="black",
            linewidth=expected_linewidth,
            alpha=expected_alpha,
        )
        ax.add_patch(box)

        # Verify the properties were set
        # Note: alpha affects both facecolor and edgecolor
        assert box.get_facecolor() == pytest.approx((1.0, 1.0, 0.0, expected_alpha))  # yellow with alpha
        assert box.get_edgecolor() == pytest.approx((0.0, 0.0, 0.0, expected_alpha))  # black with alpha
        assert box.get_linewidth() == expected_linewidth
        assert box.get_alpha() == expected_alpha
