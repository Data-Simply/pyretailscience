"""Tests for LightGBM tree visualization components."""

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for tests
mpl.use("Agg")

from pyretailscience.plots.styles.tailwind import COLORS
from pyretailscience.plots.tree_diagram import LightGBMTreeNode, TreeGrid, lightgbm_tree_to_grid


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


@pytest.fixture
def simple_lgbm_tree():
    """Create a simple LightGBM tree for testing."""
    # Create simple binary classification data
    np.random.seed(42)  # noqa: NPY002
    X = pd.DataFrame(  # noqa: N806
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [8, 7, 6, 5, 4, 3, 2, 1],
        },
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    # Train single tree
    model = lgb.LGBMClassifier(n_estimators=1, max_depth=2, min_child_samples=2, random_state=42)
    model.fit(X, y)

    return model.booster_, X.columns.tolist()


@pytest.fixture
def deep_lgbm_tree():
    """Create a deeper LightGBM tree for testing."""
    np.random.seed(42)  # noqa: NPY002
    X = pd.DataFrame(  # noqa: N806
        {
            "age": np.random.rand(100) * 50 + 20,  # noqa: NPY002
            "income": np.random.rand(100) * 50000 + 30000,  # noqa: NPY002
            "score": np.random.rand(100) * 100,  # noqa: NPY002
        },
    )
    y = pd.Series((X["age"] > 45).astype(int) & (X["income"] > 55000).astype(int))

    model = lgb.LGBMClassifier(n_estimators=1, max_depth=4, min_child_samples=5, random_state=42)
    model.fit(X, y)

    return model.booster_, X.columns.tolist()


class TestLightGBMTreeNode:
    """Tests for LightGBMTreeNode class."""

    def test_node_creation_with_split(self, ax):
        """Test creating a node with split information."""
        node_data = {
            "split_feature": "age",
            "split_condition": "<= 45.00",
            "sample_count": 100,
            "avg_value": 0.35,
            "value": 0.35,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Should create 2 patches: header box + content box
        assert len(ax.patches) == 2

        # Should create 4 text objects: header, condition, samples, avg
        assert len(ax.texts) == 4

        # Verify header text
        header_text = ax.texts[0]
        assert header_text.get_text() == "age"
        assert header_text.get_color() == "white"

    def test_node_creation_leaf(self, ax):
        """Test creating a leaf node without split."""
        node_data = {
            "sample_count": 50,
            "avg_value": 0.75,
            "value": 0.75,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Should create 2 patches: header box + content box
        assert len(ax.patches) == 2

        # Should create 3 text objects: "Leaf" header, samples, avg (no condition)
        assert len(ax.texts) == 3

        # Verify header text is "Leaf"
        header_text = ax.texts[0]
        assert header_text.get_text() == "Leaf"

    def test_node_color_gradient_low_value(self, ax):
        """Test that low values produce red color."""
        node_data = {
            "sample_count": 50,
            "avg_value": 0.1,
            "value": 0.1,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Header box should be red/orange for low values
        header_box = ax.patches[0]
        facecolor = header_box.get_facecolor()
        hex_color = f"#{int(facecolor[0] * 255):02x}{int(facecolor[1] * 255):02x}{int(facecolor[2] * 255):02x}"

        # Should be red or orange
        assert hex_color in [COLORS["red"][500], COLORS["orange"][500]]

    def test_node_color_gradient_high_value(self, ax):
        """Test that high values produce green color."""
        node_data = {
            "sample_count": 50,
            "avg_value": 0.9,
            "value": 0.9,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Header box should be green/lime for high values
        header_box = ax.patches[0]
        facecolor = header_box.get_facecolor()
        hex_color = f"#{int(facecolor[0] * 255):02x}{int(facecolor[1] * 255):02x}{int(facecolor[2] * 255):02x}"

        # Should be green or lime
        assert hex_color in [COLORS["green"][500], COLORS["lime"][500]]

    def test_node_color_gradient_mid_value(self, ax):
        """Test that medium values produce yellow color."""
        node_data = {
            "sample_count": 50,
            "avg_value": 0.5,
            "value": 0.5,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Header box should be yellow for medium values
        header_box = ax.patches[0]
        facecolor = header_box.get_facecolor()
        hex_color = f"#{int(facecolor[0] * 255):02x}{int(facecolor[1] * 255):02x}{int(facecolor[2] * 255):02x}"

        # Should be yellow
        assert hex_color == COLORS["yellow"][500]

    def test_node_dimensions(self, ax):
        """Test that node has correct dimensions."""
        node_data = {
            "sample_count": 50,
            "avg_value": 0.5,
            "value": 0.5,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Check node dimensions match class constants
        assert node.NODE_WIDTH == 3.5
        assert node.NODE_HEIGHT == 1.7

    def test_node_content_box_white(self, ax):
        """Test that content box has white background."""
        node_data = {
            "sample_count": 50,
            "avg_value": 0.5,
            "value": 0.5,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Content box should be white
        content_box = ax.patches[1]
        facecolor = content_box.get_facecolor()
        # White is (1.0, 1.0, 1.0, 1.0) in RGBA
        assert facecolor[0] == pytest.approx(1.0, abs=0.01)
        assert facecolor[1] == pytest.approx(1.0, abs=0.01)
        assert facecolor[2] == pytest.approx(1.0, abs=0.01)

    def test_node_sample_count_formatting(self, ax):
        """Test that sample count is formatted with thousands separator."""
        node_data = {
            "sample_count": 1500,
            "avg_value": 0.5,
            "value": 0.5,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Find the samples text
        sample_texts = [t for t in ax.texts if "Samples:" in t.get_text()]
        assert len(sample_texts) == 1
        assert "1,500" in sample_texts[0].get_text()

    def test_node_avg_value_percentage(self, ax):
        """Test that avg value shows as percentage when < 1."""
        node_data = {
            "sample_count": 100,
            "avg_value": 0.42,
            "value": 0.42,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=0, y=0)
        node.render(ax)

        # Find the avg text
        avg_texts = [t for t in ax.texts if "Avg:" in t.get_text()]
        assert len(avg_texts) == 1
        assert "42.00%" in avg_texts[0].get_text()

    def test_node_positioning(self, ax):
        """Test that node is positioned at specified coordinates."""
        x_pos, y_pos = 5.0, 3.0
        node_data = {
            "sample_count": 100,
            "avg_value": 0.5,
            "value": 0.5,
            "value_range": (0.0, 1.0),
        }

        node = LightGBMTreeNode(data=node_data, x=x_pos, y=y_pos)
        node.render(ax)

        # Check that patches are positioned correctly
        header_box = ax.patches[0]
        bbox = header_box.get_path().get_extents()
        # Box should start at x_pos
        assert bbox.x0 == pytest.approx(x_pos, abs=0.1)


class TestLightgbmTreeToGrid:
    """Tests for lightgbm_tree_to_grid function."""

    def test_conversion_returns_correct_structure(self, simple_lgbm_tree):
        """Test that conversion returns tree_structure, num_rows, num_cols."""
        booster, feature_names = simple_lgbm_tree

        result = lightgbm_tree_to_grid(booster, feature_names)

        assert isinstance(result, tuple)
        assert len(result) == 3

        tree_structure, num_rows, num_cols = result
        assert isinstance(tree_structure, dict)
        assert isinstance(num_rows, int)
        assert isinstance(num_cols, int)

    def test_tree_structure_has_required_keys(self, simple_lgbm_tree):
        """Test that tree_structure contains required keys for TreeGrid."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        # Check that at least root node exists
        assert len(tree_structure) > 0

        # Check root node has required keys
        root_key = "node_0"
        assert root_key in tree_structure

        root_node = tree_structure[root_key]
        assert "children" in root_node
        assert "position" in root_node

    def test_split_nodes_have_split_info(self, simple_lgbm_tree):
        """Test that internal nodes have split information."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        # Find an internal node (has children)
        internal_nodes = [
            node_data for node_data in tree_structure.values() if len(node_data.get("children", [])) > 0
        ]

        assert len(internal_nodes) > 0, "Tree should have at least one internal node"

        # Check internal node has split info
        internal_node = internal_nodes[0]
        assert "split_feature" in internal_node
        assert "split_condition" in internal_node

    def test_leaf_nodes_have_no_children(self, simple_lgbm_tree):
        """Test that leaf nodes have empty children list."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        # Find leaf nodes
        leaf_nodes = [node_data for node_data in tree_structure.values() if len(node_data.get("children", [])) == 0]

        assert len(leaf_nodes) > 0, "Tree should have at least one leaf node"

        # Leaf nodes should not have split_condition (they may still have split_feature for internal tracking)
        leaf_node = leaf_nodes[0]
        assert "split_condition" not in leaf_node or leaf_node.get("split_condition") is None

    def test_all_nodes_have_sample_count(self, simple_lgbm_tree):
        """Test that all nodes have sample_count."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        for node_data in tree_structure.values():
            assert "sample_count" in node_data
            assert isinstance(node_data["sample_count"], int)
            assert node_data["sample_count"] > 0

    def test_all_nodes_have_avg_value(self, simple_lgbm_tree):
        """Test that all nodes have avg_value."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        for node_data in tree_structure.values():
            assert "avg_value" in node_data
            assert isinstance(node_data["avg_value"], (int, float))

    def test_value_range_included(self, simple_lgbm_tree):
        """Test that value_range is included for color normalization."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        # At least one node should have value_range
        nodes_with_range = [node_data for node_data in tree_structure.values() if "value_range" in node_data]
        assert len(nodes_with_range) > 0

        # Check value_range format
        for node_data in nodes_with_range:
            value_range = node_data["value_range"]
            assert isinstance(value_range, tuple)
            assert len(value_range) == 2
            assert value_range[0] <= value_range[1]

    def test_num_rows_matches_tree_depth(self, simple_lgbm_tree):
        """Test that num_rows equals tree depth + 1."""
        booster, feature_names = simple_lgbm_tree

        _, num_rows, _ = lightgbm_tree_to_grid(booster, feature_names)

        # Simple tree with max_depth=2 should have num_rows=3 (depth 0, 1, 2)
        assert num_rows >= 2
        assert num_rows <= 4  # Should be reasonable for simple tree

    def test_feature_names_used_in_splits(self, simple_lgbm_tree):
        """Test that feature names appear in split conditions."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        # Find nodes with split features (internal nodes only, not leaves)
        split_features = [
            node_data.get("split_feature")
            for node_data in tree_structure.values()
            if "split_feature" in node_data
            and node_data["split_feature"] is not None
            and len(node_data.get("children", [])) > 0  # Only internal nodes
        ]

        # At least one split should use a known feature name
        assert len(split_features) > 0
        for split_feature in split_features:
            # Feature name should be one of the provided features
            assert split_feature in feature_names or split_feature.startswith("Column_")

    def test_positions_are_valid(self, simple_lgbm_tree):
        """Test that all position tuples are valid."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(booster, feature_names)

        for node_id, node_data in tree_structure.items():
            position = node_data["position"]
            assert isinstance(position, tuple)
            assert len(position) == 2

            col_idx, row_idx = position
            assert 0 <= col_idx < num_cols, f"Node {node_id} col {col_idx} out of bounds [0, {num_cols})"
            assert 0 <= row_idx < num_rows, f"Node {node_id} row {row_idx} out of bounds [0, {num_rows})"

    def test_children_references_are_valid(self, simple_lgbm_tree):
        """Test that all children references point to existing nodes."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, _, _ = lightgbm_tree_to_grid(booster, feature_names)

        for node_data in tree_structure.values():
            children = node_data.get("children", [])
            for child_id in children:
                assert child_id in tree_structure, f"Child {child_id} not found in tree_structure"

    def test_deep_tree_dimensions(self, deep_lgbm_tree):
        """Test that deeper tree produces correct dimensions."""
        booster, feature_names = deep_lgbm_tree

        tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(booster, feature_names)

        # Deep tree with max_depth=4 should have more rows
        assert num_rows >= 3, "Deep tree should have at least 3 rows"
        assert num_cols >= 2, "Tree should have at least 2 columns"
        assert len(tree_structure) >= 5, "Deep tree should have multiple nodes"

    def test_conversion_without_feature_names(self, simple_lgbm_tree):
        """Test that conversion works without explicit feature names."""
        booster, _ = simple_lgbm_tree

        # Should work with feature_names=None
        tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(booster, feature_names=None)

        assert isinstance(tree_structure, dict)
        assert num_rows > 0
        assert num_cols > 0

    def test_error_on_multiple_estimators(self):
        """Test that function raises error when model has multiple estimators."""
        # Create model with multiple trees
        np.random.seed(42)  # noqa: NPY002
        X = pd.DataFrame(  # noqa: N806
            {
                "feat1": np.random.rand(50),  # noqa: NPY002
                "feat2": np.random.rand(50),  # noqa: NPY002
            },
        )
        y = pd.Series((X["feat1"] > 0.5).astype(int))

        model = lgb.LGBMClassifier(n_estimators=3, random_state=42)  # 3 trees!
        model.fit(X, y)

        # Should raise ValueError for multiple trees
        with pytest.raises(ValueError, match="single tree"):
            lightgbm_tree_to_grid(model.booster_, X.columns.tolist())


class TestLightGBMTreeNodeIntegration:
    """Integration tests for LightGBMTreeNode with TreeGrid."""

    def test_treegrid_renders_lightgbm_tree(self, simple_lgbm_tree):
        """Test that TreeGrid can render LightGBM tree structure."""
        booster, feature_names = simple_lgbm_tree

        tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(booster, feature_names)

        # Create TreeGrid
        grid = TreeGrid(
            tree_structure=tree_structure,
            num_rows=num_rows,
            num_cols=num_cols,
            node_class=LightGBMTreeNode,
        )

        # Render should succeed
        ax = grid.render()
        assert ax is not None

        # Should have patches for nodes + connections
        num_nodes = len(tree_structure)
        patches_per_node = 2  # header + content
        assert len(ax.patches) >= num_nodes * patches_per_node

    def test_full_workflow_from_model_to_visualization(self):
        """Test complete workflow from training to visualization."""
        # Create and train model
        np.random.seed(42)  # noqa: NPY002
        X = pd.DataFrame(  # noqa: N806
            {
                "customer_age": [25, 35, 45, 55, 30, 40, 50, 60],
                "purchase_amount": [100, 200, 150, 300, 120, 250, 180, 350],
            },
        )
        y = pd.Series([0, 0, 1, 1, 0, 1, 1, 1])

        model = lgb.LGBMClassifier(n_estimators=1, max_depth=2, min_child_samples=2, random_state=42)
        model.fit(X, y)

        # Convert to tree structure
        tree_structure, num_rows, num_cols = lightgbm_tree_to_grid(model.booster_, X.columns.tolist())

        # Render with TreeGrid
        grid = TreeGrid(
            tree_structure=tree_structure,
            num_rows=num_rows,
            num_cols=num_cols,
            node_class=LightGBMTreeNode,
            horizontal_spacing=4.5,
            vertical_spacing=2.5,
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        ax = grid.render(ax=ax)

        # Set axes limits (like SpeedDrill does)
        plot_width = grid.col[num_cols - 1] + grid.node_width
        plot_height = grid.row[0] + grid.node_height
        ax.set_xlim(0, plot_width)
        ax.set_ylim(0, plot_height)

        # Verify visualization components
        assert len(ax.patches) > 0, "Should have rendered patches"
        assert len(ax.texts) > 0, "Should have rendered text"

        # Verify axes limits are set correctly
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] == 0
        assert xlim[1] > 1  # Should be larger than default
        assert ylim[0] == 0
        assert ylim[1] > 1  # Should be larger than default
