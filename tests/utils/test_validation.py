"""Tests for pyretailscience.utils.validation."""

import ibis
import pandas as pd
import pytest

from pyretailscience.utils.validation import validate_columns


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
