"""Geospatial Distance Analysis for Retail Location Intelligence.

This module provides efficient geospatial distance calculations that power location-based retail analytics and
strategic decision-making.

## Technical Features

- **Ibis-Based Computation**: Scalable processing within existing data pipelines
- **Database Integration**: Calculations performed in SQL databases for efficiency
- **Haversine Formula**: Accurate great-circle distance computation
- **Backend Agnostic**: Works with multiple database and processing engines

## Limitations

- **Spherical Earth Assumption**: Minor inaccuracies due to Earth's actual oblate shape
- **Straight-Line Distance**: Measures "as the crow flies", not driving distances
- **Requires Trigonometric Functions**: Backend must support mathematical functions
"""

import ibis


def haversine_distance(
    lat_col: ibis.expr.types.Column,
    lon_col: ibis.expr.types.Column,
    target_lat_col: ibis.expr.types.Column,
    target_lon_col: ibis.expr.types.Column,
    radius: float = 6371.0,
) -> ibis.expr.types.Column:
    """Computes the Haversine distance between two sets of latitude and longitude columns.

    Parameters:
        lat_col (ibis.expr.types.Column): Column containing source latitudes.
        lon_col (ibis.expr.types.Column): Column containing source longitudes.
        target_lat_col (ibis.expr.types.Column): Column containing target latitudes.
        target_lon_col (ibis.expr.types.Column): Column containing target longitudes.
        radius (float, optional): Earth's radius in kilometers (default: 6371 km).

    Returns:
        ibis.expr.types.Column: An Ibis expression representing the computed distances.
    """
    lat1_rad = lat_col.radians()
    lat2_rad = target_lat_col.radians()
    delta_lat = (target_lat_col - lat_col).radians()
    delta_lon = (target_lon_col - lon_col).radians()

    a = (delta_lat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2).sin().pow(2)
    c = 2 * a.sqrt().asin()

    return radius * c
