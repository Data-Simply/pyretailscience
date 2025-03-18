"""Tests for the haversine distance module."""

import ibis
import pandas as pd
import pytest

from pyretailscience.analysis.haversine import haversine_distance


@pytest.fixture
def sample_ibis_table():
    """Fixture to provide a sample Ibis table for testing."""
    data = {
        "lat1": [37.7749, 34.0522],
        "lon1": [-122.4194, -118.2437],
        "lat2": [40.7128, 36.1699],
        "lon2": [-74.0060, -115.1398],
    }
    df = pd.DataFrame(data)
    return ibis.memtable(df)


def test_haversine_distance(sample_ibis_table):
    """Test the haversine_distance function for correct distance calculation."""
    t = sample_ibis_table
    distance_expr = haversine_distance(t["lat1"], t["lon1"], t["lat2"], t["lon2"])

    assert isinstance(distance_expr, ibis.expr.types.Column), "Output should be an Ibis expression."

    result_df = t.mutate(distance=distance_expr).execute()

    expected_distances = [4129.086165, 367.606322]

    for i, expected in enumerate(expected_distances):
        assert pytest.approx(result_df.iloc[i]["distance"], rel=1e-3) == expected, f"Row {i} distance mismatch."
