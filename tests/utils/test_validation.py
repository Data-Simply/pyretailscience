"""Tests for pyretailscience.utils.validation."""

import ibis
import pandas as pd
import pytest

from pyretailscience.utils.validation import ensure_ibis_table, validate_columns


class TestValidateColumns:
    """Tests for the validate_columns utility function."""

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_raises_when_column_missing(self, input_type):
        """Test validation raises ValueError listing missing columns for both input types."""
        pdf = pd.DataFrame({"customer_id": [1, 2], "store_id": [101, 102]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        with pytest.raises(ValueError, match="unit_spend"):
            validate_columns(df, ["customer_id", "unit_spend"])

    def test_error_message_lists_all_missing_columns_in_sorted_order(self):
        """Test that the error message lists missing columns in sorted order."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(ValueError, match=r"\['store_id', 'unit_spend'\]"):
            validate_columns(df, ["customer_id", "unit_spend", "store_id"])

    @pytest.mark.parametrize("required_cols", ["unit_spend", None, 42, ("unit_spend",)])
    def test_raises_type_error_when_required_cols_is_not_a_list(self, required_cols):
        """Test that passing a non-list for required_cols raises TypeError."""
        df = pd.DataFrame({"customer_id": [1], "unit_spend": [5.0]})
        with pytest.raises(TypeError, match="required_cols must be a list of column names"):
            validate_columns(df, required_cols)

    def test_does_not_raise_with_superset_of_columns(self):
        """Test validation succeeds when DataFrame has more columns than required."""
        df = pd.DataFrame({"customer_id": [1], "unit_spend": [5.0], "store_id": [10]})
        result = validate_columns(df, ["customer_id"])
        assert result is None


class TestEnsureIbisTable:
    """Tests for the ensure_ibis_table helper function."""

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"customer_id": [1, 2, 3], "unit_spend": [4.50, 5.99, 6.00]}),
            pd.DataFrame({"customer_id": pd.Series([], dtype="int64"), "unit_spend": pd.Series([], dtype="float64")}),
        ],
        ids=["non_empty", "empty"],
    )
    def test_pandas_dataframe_converts_to_ibis_table(self, df):
        """Test that pandas DataFrame is converted to ibis Table with data preserved."""
        result = ensure_ibis_table(df)
        assert isinstance(result, ibis.Table)
        result_df = result.execute()
        pd.testing.assert_frame_equal(result_df, df)

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"customer_id": [1, 2, 3], "unit_spend": [4.50, 5.99, 6.00]}),
            pd.DataFrame({"customer_id": pd.Series([], dtype="int64"), "unit_spend": pd.Series([], dtype="float64")}),
        ],
        ids=["non_empty", "empty"],
    )
    def test_ibis_table_returns_unchanged(self, df):
        """Test that ibis Table input is returned unchanged."""
        ibis_table = ibis.memtable(df)
        result = ensure_ibis_table(ibis_table)
        assert result is ibis_table

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
