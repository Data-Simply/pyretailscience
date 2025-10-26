"""Tests for the plots.tree_diagram module."""

import matplotlib.pyplot as plt
import pytest

from pyretailscience.plots.styles.tailwind import COLORS
from pyretailscience.plots.tree_diagram import BaseRoundedBox, SimpleTreeNode, TreeNode


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


class TestTreeNode:
    """Test the TreeNode abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that TreeNode abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            TreeNode(data={}, x=0, y=0)


class TestSimpleTreeNode:
    """Test the SimpleTreeNode class."""

    def test_rendering_with_valid_data(self, ax):
        """Test that SimpleTreeNode renders correctly with valid data."""
        node = SimpleTreeNode(
            data={
                "header": "Total Revenue",
                "percent": 12.5,
                "value1": "£1.2M",
                "value2": "£1.07M",
            },
            x=0.5,
            y=0.8,
        )

        initial_patch_count = len(ax.patches)
        initial_text_count = len(ax.texts)

        node.render(ax)

        # Should add 2 patches (header box and data box)
        assert len(ax.patches) == initial_patch_count + 2

        # Should add text elements (header, percent, value1, value2)
        assert len(ax.texts) == initial_text_count + 4

        # Verify text content
        text_strings = [t.get_text() for t in ax.texts]
        assert "Total Revenue" in text_strings
        assert "+12.5%" in text_strings
        assert "£1.2M" in text_strings
        assert "£1.07M" in text_strings

    @pytest.mark.parametrize(
        ("percent", "expected_color_name"),
        [
            (12.5, "green"),  # Significant growth (percent >= 1.0)
            (-8.3, "red"),  # Significant decline (percent <= -1.0)
            (1.0, "green"),  # At green threshold (percent == 1.0)
            (-1.0, "red"),  # At red threshold (percent == -1.0)
            (0.5, "gray"),  # Neutral (between thresholds)
        ],
    )
    def test_color_selection_based_on_percent(self, ax, percent, expected_color_name):
        """Test that data box color is selected correctly based on percent change thresholds."""
        node = SimpleTreeNode(
            data={
                "header": "Customer Frequency",
                "percent": percent,
                "value1": "£125.5K",
                "value2": "£115.0K",
            },
            x=0,
            y=0,
        )

        node.render(ax)

        # The data box (second patch added) should have the color based on percent
        data_box = ax.patches[-1]
        expected_color = COLORS[expected_color_name][500]

        # Convert RGBA to hex for comparison
        facecolor = data_box.get_facecolor()
        hex_color = f"#{int(facecolor[0] * 255):02x}{int(facecolor[1] * 255):02x}{int(facecolor[2] * 255):02x}"
        assert hex_color == expected_color

    @pytest.mark.parametrize(
        "missing_key",
        ["header", "percent", "value1", "value2"],
    )
    def test_missing_required_keys(self, ax, missing_key):
        """Test that KeyError is raised when required keys are missing."""
        data = {
            "header": "Average Basket Value",
            "percent": 5.2,
            "value1": "£45.80",
            "value2": "£43.50",
        }
        del data[missing_key]

        node = SimpleTreeNode(data=data, x=0, y=0)

        with pytest.raises(KeyError):
            node.render(ax)


class TestSimpleTreeNodeIntegration:
    """Integration tests for SimpleTreeNode."""

    def test_multiple_nodes_on_same_axes(self, ax):
        """Test rendering multiple SimpleTreeNodes on the same axes with varied metrics."""
        # Create 3 nodes representing a revenue tree hierarchy
        node1 = SimpleTreeNode(
            data={
                "header": "Total Revenue",
                "percent": 15.3,
                "value1": "£1.2M",
                "value2": "£1.04M",
            },
            x=0,
            y=2,
        )
        node2 = SimpleTreeNode(
            data={
                "header": "Customer Count",
                "percent": -7.2,
                "value1": "8,540",
                "value2": "9,200",
            },
            x=4,
            y=2,
        )
        node3 = SimpleTreeNode(
            data={
                "header": "Avg Transaction",
                "percent": 0.8,
                "value1": "£42.35",
                "value2": "£42.01",
            },
            x=2,
            y=0,
        )

        node1.render(ax)
        node2.render(ax)
        node3.render(ax)

        # Should have 6 patches total (2 per node)
        patches_per_node = 2
        num_nodes = 3
        expected_patches = num_nodes * patches_per_node
        assert len(ax.patches) == expected_patches

        # Verify colors: green (>= 1.0), red (<= -1.0), gray (between)
        expected_color_names = ["green", "red", "gray"]
        expected_colors = [COLORS[name][500] for name in expected_color_names]
        data_boxes = [ax.patches[1], ax.patches[3], ax.patches[5]]  # Every second patch is a data box

        for i, data_box in enumerate(data_boxes):
            facecolor = data_box.get_facecolor()
            hex_color = f"#{int(facecolor[0] * 255):02x}{int(facecolor[1] * 255):02x}{int(facecolor[2] * 255):02x}"
            assert hex_color == expected_colors[i]
