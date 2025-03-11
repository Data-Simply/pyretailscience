"""Tests for the haversine distance module."""
import pandas as pd
import pytest

from pyretailscience.analysis.haversine import haversine_distance


@pytest.fixture()
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "lat1": [37.7749, 34.0522],
        "lon1": [-122.4194, -118.2437],
        "lat2": [40.7128, 36.1699],
        "lon2": [-74.0060, -115.1398],
    }
    return pd.DataFrame(data)


def test_haversine_distance(sample_dataframe):
    """Test the haversine_distance function for correct distance calculation."""
    result_df = haversine_distance(sample_dataframe, "lat1", "lon1", "lat2", "lon2")

    assert "distance" in result_df.columns, "Output DataFrame should contain a 'distance' column."
    assert result_df.shape[0] == sample_dataframe.shape[0], "Output DataFrame should have the same number of rows."

    expected_distances = [4129.086165, 367.606322]

    for i, expected in enumerate(expected_distances):
        assert pytest.approx(result_df.iloc[i]["distance"], rel=1e-3) == expected, f"Row {i} distance mismatch."
