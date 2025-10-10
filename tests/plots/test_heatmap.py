"""Tests for the heatmap plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.plots import heatmap
from pyretailscience.plots.styles import graph_utils as gu

RNG = np.random.default_rng(42)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_migration_dataframe():
    """Generates a sample migration matrix DataFrame."""
    data = np.round(RNG.uniform(0, 1, size=(5, 5)), 2)
    return pd.DataFrame(
        data,
        columns=[f"Quintile {i + 1}" for i in range(5)],
        index=[f"Previous Q{i + 1}" for i in range(5)],
    )


@pytest.fixture
def sample_correlation_dataframe():
    """Generates a sample correlation matrix DataFrame."""
    # Create symmetric correlation matrix
    data = RNG.uniform(-1, 1, size=(4, 4))
    data = (data + data.T) / 2  # Make symmetric
    np.fill_diagonal(data, 1.0)  # Diagonal should be 1
    return pd.DataFrame(
        np.round(data, 2),
        columns=["Feature A", "Feature B", "Feature C", "Feature D"],
        index=["Feature A", "Feature B", "Feature C", "Feature D"],
    )


@pytest.fixture
def sample_large_dataframe():
    """Generates a larger sample DataFrame for testing."""
    data = np.round(RNG.uniform(0, 100, size=(10, 8)), 1)
    return pd.DataFrame(data, columns=[f"Col {i + 1}" for i in range(8)], index=[f"Row {i + 1}" for i in range(10)])


@pytest.fixture
def sample_single_row_dataframe():
    """Generates a single row DataFrame for edge case testing."""
    data = np.round(RNG.uniform(0, 1, size=(1, 6)), 2)
    return pd.DataFrame(data, columns=[f"Month {i + 1}" for i in range(6)], index=["Single Row"])


@pytest.fixture
def sample_single_column_dataframe():
    """Generates a single column DataFrame for edge case testing."""
    data = np.round(RNG.uniform(0, 1, size=(6, 1)), 2)
    return pd.DataFrame(data, columns=["Single Col"], index=[f"Row {i + 1}" for i in range(6)])


@pytest.fixture
def sample_dataframe_with_negatives():
    """Generates a DataFrame with negative values."""
    data = np.round(RNG.uniform(-1, 1, size=(4, 4)), 2)
    return pd.DataFrame(data, columns=[f"Col {i + 1}" for i in range(4)], index=[f"Row {i + 1}" for i in range(4)])


@pytest.fixture
def _mock_gu_functions(mocker):
    """Mocks graph utility functions to avoid modifying global styles."""
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_graph_styles", side_effect=lambda ax, **kwargs: ax)
    mocker.patch("pyretailscience.plots.styles.graph_utils.standard_tick_styles", side_effect=lambda ax: ax)
    mocker.patch(
        "pyretailscience.plots.styles.graph_utils.add_source_text",
        side_effect=lambda ax, source_text: ax,
    )


class TestHeatmapBasicFunctionality:
    """Test basic heatmap functionality."""

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_basic_heatmap(self, sample_migration_dataframe):
        """Test basic heatmap creation."""
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
            x_label="Current Quintile",
            y_label="Previous Quintile",
            title="Customer Migration Matrix",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_correlation_matrix(self, sample_correlation_dataframe):
        """Test heatmap with correlation data (including negative values)."""
        result_ax = heatmap.plot(
            df=sample_correlation_dataframe,
            cbar_label="Correlation",
            title="Feature Correlations",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_large_values(self, sample_large_dataframe):
        """Test heatmap with larger numerical values."""
        result_ax = heatmap.plot(
            df=sample_large_dataframe,
            cbar_label="Count",
            title="Large Values Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_negative_values(self, sample_dataframe_with_negatives):
        """Test heatmap with negative values."""
        result_ax = heatmap.plot(
            df=sample_dataframe_with_negatives,
            cbar_label="Value",
            title="Negative Values Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0


class TestHeatmapParameters:
    """Test different parameter combinations."""

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_source_text(self, sample_migration_dataframe):
        """Test heatmap with source text annotation."""
        source_text = "Source: Test Migration Data"

        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
            source_text=source_text,
        )

        gu.add_source_text.assert_called_once_with(ax=result_ax, source_text=source_text)

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_ax_none(self, sample_migration_dataframe):
        """Test heatmap when ax is None (should create a new figure)."""
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
            ax=None,
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0
        assert result_ax.figure is not None

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_figsize(self, sample_migration_dataframe):
        """Test heatmap with a specified figsize."""
        width = 12
        height = 8
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
            figsize=(width, height),
        )

        assert isinstance(result_ax, Axes)
        assert result_ax.figure.get_size_inches()[0] == width
        assert result_ax.figure.get_size_inches()[1] == height

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_all_optional_labels(self, sample_migration_dataframe):
        """Test heatmap with all optional labels provided."""
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
            x_label="Current Quintile",
            y_label="Previous Quintile",
            title="Customer Migration Matrix",
        )

        assert isinstance(result_ax, Axes)
        # Verify standard_graph_styles was called with the correct parameters
        gu.standard_graph_styles.assert_called_once_with(
            ax=result_ax,
            title="Customer Migration Matrix",
            x_label="Current Quintile",
            y_label="Previous Quintile",
        )

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_with_kwargs(self, sample_migration_dataframe):
        """Test heatmap with additional kwargs passed to imshow."""
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
            alpha=0.8,
            interpolation="nearest",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0


class TestHeatmapEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_single_row(self, sample_single_row_dataframe):
        """Test heatmap with single row DataFrame."""
        result_ax = heatmap.plot(
            df=sample_single_row_dataframe,
            cbar_label="Value",
            title="Single Row Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_single_column(self, sample_single_column_dataframe):
        """Test heatmap with single column DataFrame."""
        result_ax = heatmap.plot(
            df=sample_single_column_dataframe,
            cbar_label="Value",
            title="Single Column Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_plot_single_cell(self):
        """Test heatmap with single cell DataFrame."""
        single_cell_df = pd.DataFrame([[0.5]], columns=["Col"], index=["Row"])

        result_ax = heatmap.plot(
            df=single_cell_df,
            cbar_label="Value",
            title="Single Cell Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    def test_plot_empty_dataframe(self):
        """Test heatmap with empty DataFrame should raise an error."""
        empty_df = pd.DataFrame()

        with pytest.raises((ValueError, IndexError)):
            heatmap.plot(
                df=empty_df,
                cbar_label="Value",
            )


class TestHeatmapVisualElements:
    """Test visual elements and text formatting."""

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_text_displays_raw_values(self, sample_migration_dataframe):
        """Test that cell text displays raw values (not percentages)."""
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
        )

        # Check that text elements exist
        texts = result_ax.texts
        assert len(texts) > 0

        # Check that text format matches expected decimal format
        for text in texts:
            text_value = text.get_text()
            # Should be in format like "0.42", not "42%"
            assert "%" not in text_value
            # Should be able to parse as float
            float(text_value)

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_colorbar_exists(self, sample_migration_dataframe):
        """Test that colorbar is created with correct label."""
        cbar_label = "Test Colorbar Label"
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label=cbar_label,
        )

        # Check that figure has multiple axes (plot + colorbar)
        result_length = 2
        assert len(result_ax.figure.axes) >= result_length

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_grid_styling(self, sample_migration_dataframe):
        """Test that grid styling is applied correctly."""
        result_ax = heatmap.plot(
            df=sample_migration_dataframe,
            cbar_label="Migration Rate",
        )

        # Check that minor ticks are set for grid lines
        minor_xticks = result_ax.get_xticks(minor=True)
        minor_yticks = result_ax.get_yticks(minor=True)

        expected_x_ticks = len(sample_migration_dataframe.columns) + 1
        expected_y_ticks = len(sample_migration_dataframe.index) + 1

        assert len(minor_xticks) == expected_x_ticks
        assert len(minor_yticks) == expected_y_ticks


class TestHeatmapDataTypes:
    """Test with different data types and value ranges."""

    @pytest.mark.parametrize(
        "value_range",
        [
            (0, 1),  # 0-1 range (like percentages)
            (0, 100),  # 0-100 range
            (-50, 50),  # Negative to positive
            (1000, 2000),  # Large numbers
        ],
    )
    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_different_value_ranges(self, value_range):
        """Test heatmap with different value ranges."""
        min_val, max_val = value_range
        data = np.round(RNG.uniform(min_val, max_val, size=(3, 3)), 2)
        df = pd.DataFrame(data, columns=[f"Col {i}" for i in range(3)], index=[f"Row {i}" for i in range(3)])

        result_ax = heatmap.plot(
            df=df,
            cbar_label="Value",
            title=f"Range {min_val}-{max_val}",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_integer_data(self):
        """Test heatmap with integer data."""
        data = RNG.integers(0, 100, size=(4, 4))
        df = pd.DataFrame(data, columns=[f"Col {i}" for i in range(4)], index=[f"Row {i}" for i in range(4)])

        result_ax = heatmap.plot(
            df=df,
            cbar_label="Count",
            title="Integer Data Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0

    @pytest.mark.usefixtures("_mock_gu_functions")
    def test_with_nan_values(self):
        """Test heatmap behavior with NaN values."""
        data = np.round(RNG.uniform(0, 1, size=(3, 3)), 2)
        data[1, 1] = np.nan  # Insert a NaN value
        df = pd.DataFrame(data, columns=[f"Col {i}" for i in range(3)], index=[f"Row {i}" for i in range(3)])

        # Should handle NaN values gracefully
        result_ax = heatmap.plot(
            df=df,
            cbar_label="Value",
            title="NaN Values Test",
        )

        assert isinstance(result_ax, Axes)
        assert len(result_ax.get_children()) > 0
