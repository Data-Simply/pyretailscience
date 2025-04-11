"""This module provides functionality for computing geospatial distances using Ibis expressions.

It defines functions for efficient geospatial analysis on structured data tables, leveraging
Ibis for optimized query execution.

### Core Features

- **Ibis-Based Computation**: Uses Ibis expressions for scalable processing.
- **Haversine Distance Calculation**: Computes great-circle distances dynamically as an Ibis expression.
- **Backend Agnostic**: Works with multiple Ibis-supported backends, including SQL-based databases.
- **Efficient Query Optimization**: Defers computation to the database or processing engine.

### Use Cases

- **Geospatial Filtering**: Identify locations within a certain radius using database queries.
- **Spatial Analysis**: Analyze movement patterns and distances between geographic points.
- **Logistics & Routing**: Optimize route planning by calculating distances dynamically.

### Limitations and Warnings

- **Requires Ibis-Compatible Backend**: Ensure your Ibis backend supports trigonometric functions.
- **Assumes Spherical Earth**: Uses the Haversine formula, which introduces slight inaccuracies due to Earth's oblate shape.
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
