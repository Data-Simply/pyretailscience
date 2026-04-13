"""Tests for pyretailscience.metrics.base."""

import ibis
import numpy as np
import pandas as pd
import pytest

from pyretailscience.metrics.base import ratio_metric
from pyretailscience.utils.validation import ensure_ibis_table


class TestRatioMetric:
    """Tests for the ratio_metric helper function."""

    def test_operator_precedence_division_before_scale(self):
        """Test that division occurs before multiplication by scale (not after)."""
        # Create a simple ibis expression: 10 / 4 * 100 should be 250.0
        # If precedence were wrong (10 / (4 * 100)), result would be 0.025
        numerator = ibis.literal(10)
        denominator = ibis.literal(4)
        result = ratio_metric(numerator, denominator, scale=100).execute()
        expected = 250.0  # (10 / 4) * 100 = 2.5 * 100
        assert result == expected

    def test_basic_ratio_with_default_scale(self):
        """Test basic ratio calculation with default percentage scale."""
        numerator = ibis.literal(1)
        denominator = ibis.literal(2)
        result = ratio_metric(numerator, denominator).execute()
        expected = 50.0
        assert result == expected

    def test_ratio_with_custom_scale(self):
        """Test ratio calculation with custom scale factor."""
        numerator = ibis.literal(3)
        denominator = ibis.literal(4)
        result = ratio_metric(numerator, denominator, scale=1000).execute()
        expected = 750.0
        assert result == expected

    def test_zero_denominator_returns_nan(self):
        """Test that zero denominator returns NaN (NULL in ibis)."""
        numerator = ibis.literal(10)
        denominator = ibis.literal(0)
        result = ratio_metric(numerator, denominator).execute()
        assert np.isnan(result)

    def test_ratio_with_table_columns(self):
        """Test ratio_metric with actual table columns from realistic data."""
        df = pd.DataFrame(
            {
                "stores_selling": [2, 1, 4],
                "total_stores": [4, 4, 4],
            }
        )
        table = ibis.memtable(df)
        result_table = table.mutate(pct=ratio_metric(table.stores_selling, table.total_stores))
        result = result_table.execute()
        expected_pct = [50.0, 25.0, 100.0]
        pd.testing.assert_series_equal(
            result["pct"],
            pd.Series(expected_pct, name="pct"),
        )

    def test_multiple_zero_denominators_in_column(self):
        """Test handling of mixed zero and non-zero denominators in a column."""
        df = pd.DataFrame(
            {
                "numerator": [10, 20, 30],
                "denominator": [2, 0, 5],
            }
        )
        table = ibis.memtable(df)
        result_table = table.mutate(ratio=ratio_metric(table.numerator, table.denominator, scale=1))
        result = result_table.execute()
        expected_first = 5.0  # 10/2
        expected_last = 6.0  # 30/5
        assert result["ratio"].iloc[0] == expected_first
        assert np.isnan(result["ratio"].iloc[1])  # 20/0 -> NaN
        assert result["ratio"].iloc[2] == expected_last


class TestEnsureIbisTable:
    """Tests for the ensure_ibis_table helper function."""

    def test_pandas_dataframe_converts_to_ibis_table(self):
        """Test that pandas DataFrame is converted to ibis Table."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = ensure_ibis_table(df)
        assert isinstance(result, ibis.Table)
        # Verify the data is preserved
        result_df = result.execute()
        pd.testing.assert_frame_equal(result_df, df)

    def test_ibis_table_returns_unchanged(self):
        """Test that ibis Table input is returned unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ibis_table = ibis.memtable(df)
        result = ensure_ibis_table(ibis_table)
        assert result is ibis_table  # Should be the same object

    @pytest.mark.parametrize(
        "invalid_input",
        [
            {"a": [1, 2, 3]},  # dict
            [1, 2, 3],  # list
            None,  # None
            "not a dataframe",  # string
            42,  # int
        ],
        ids=["dict", "list", "None", "string", "int"],
    )
    def test_invalid_input_raises_type_error(self, invalid_input):
        """Test that invalid input types raise TypeError with correct message."""
        with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an Ibis Table"):
            ensure_ibis_table(invalid_input)

    def test_empty_pandas_dataframe_converts(self):
        """Test that empty pandas DataFrame converts successfully."""
        df = pd.DataFrame()
        result = ensure_ibis_table(df)
        assert isinstance(result, ibis.Table)

    def test_empty_ibis_table_returns_unchanged(self):
        """Test that empty ibis Table is returned unchanged."""
        df = pd.DataFrame()
        ibis_table = ibis.memtable(df)
        result = ensure_ibis_table(ibis_table)
        assert result is ibis_table
