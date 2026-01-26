"""Unified integration tests for haversine distance function with multiple database backends."""

import ibis
import pytest

from pyretailscience.analysis.haversine import haversine_distance


def test_haversine_integration():
    """Integration test for haversine distance function.

    This test doesn't use the parameterized transactions_table since
    the haversine function works with literal values rather than
    database tables. The test verifies the function works consistently
    across different Ibis backends.
    """
    lat1 = ibis.literal(37.7749, type="float64")
    lon1 = ibis.literal(-122.4194, type="float64")
    lat2 = ibis.literal(40.7128, type="float64")
    lon2 = ibis.literal(-74.0060, type="float64")

    distance_expr = haversine_distance(lat1, lon1, lat2, lon2)
    result = distance_expr.execute()

    expected_distance = 4129.086165
    assert pytest.approx(result, rel=1e-3) == expected_distance, "Distance calculation error"
