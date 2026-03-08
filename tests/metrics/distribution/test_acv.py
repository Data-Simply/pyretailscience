"""Tests for pyretailscience.metrics.distribution.acv."""

import ibis
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyretailscience.metrics.distribution.acv import Acv


class TestAcv:
    """Tests for the Acv metric class."""

    def test_acv_total_no_grouping(self):
        """Test total ACV across all transactions without grouping."""
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 1, 2],
                "store_id": [101, 101, 102, 102, 103],
                "product_id": [10, 20, 30, 40, 50],
                "unit_spend": [500_000.0, 750_000.0, 300_000.0, 600_000.0, 350_000.0],
            }
        )
        result = Acv(df).df
        expected = pd.DataFrame({"acv": [2.5]})
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_acv_grouped_by_store(self, input_type):
        """Test ACV grouped by store returns correct per-store values for both input types."""
        pdf = pd.DataFrame(
            {
                "store_id": [101, 101, 102, 102, 103],
                "unit_spend": [400_000.0, 600_000.0, 300_000.0, 200_000.0, 500_000.0],
            }
        )
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        result = Acv(df, group_by="store_id").df.sort_values("store_id").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "store_id": [101, 102, 103],
                "acv": [1.0, 0.5, 0.5],
            }
        )
        assert_frame_equal(result, expected)

    def test_acv_group_by_list(self):
        """Test ACV grouped by multiple columns."""
        df = pd.DataFrame(
            {
                "store_id": [101, 101, 102],
                "region": ["North", "North", "South"],
                "unit_spend": [1_000_000.0, 500_000.0, 2_000_000.0],
            }
        )
        result = Acv(df, group_by=["store_id", "region"]).df.sort_values("store_id").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "store_id": [101, 102],
                "region": ["North", "South"],
                "acv": [1.5, 2.0],
            }
        )
        assert_frame_equal(result, expected)

    def test_acv_with_nan_values(self):
        """Test that NaN values are excluded from the ACV sum."""
        df = pd.DataFrame(
            {
                "store_id": [101, 101, 102],
                "unit_spend": [1_000_000.0, np.nan, 500_000.0],
            }
        )
        result = Acv(df, group_by="store_id").df.sort_values("store_id").reset_index(drop=True)
        expected = pd.DataFrame(
            {
                "store_id": [101, 102],
                "acv": [1.0, 0.5],
            }
        )
        assert_frame_equal(result, expected)

    def test_acv_missing_column_raises(self):
        """Test that missing unit_spend column raises ValueError."""
        df = pd.DataFrame({"customer_id": [1, 2], "store_id": [101, 102]})
        with pytest.raises(ValueError, match="missing"):
            Acv(df)

    def test_acv_missing_group_by_column_raises(self):
        """Test that missing group_by column raises ValueError."""
        df = pd.DataFrame({"unit_spend": [100.0, 200.0]})
        with pytest.raises(ValueError, match="missing"):
            Acv(df, group_by="store_id")

    def test_acv_custom_scale_factor(self):
        """Test ACV with a custom scale factor."""
        df = pd.DataFrame(
            {
                "store_id": [101, 102],
                "unit_spend": [5_000.0, 10_000.0],
            }
        )
        result = Acv(df, acv_scale_factor=1_000).df
        expected = pd.DataFrame({"acv": [15.0]})
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("scale_factor", [0, -1_000])
    def test_acv_non_positive_scale_factor_raises(self, scale_factor):
        """Test that zero or negative acv_scale_factor raises ValueError."""
        df = pd.DataFrame({"unit_spend": [500_000.0, 1_000_000.0]})
        with pytest.raises(ValueError, match="acv_scale_factor must be positive"):
            Acv(df, acv_scale_factor=scale_factor)

    def test_acv_invalid_type_raises(self):
        """Test that passing a non-DataFrame/Table raises TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame or an Ibis Table"):
            Acv({"unit_spend": [100.0]})
