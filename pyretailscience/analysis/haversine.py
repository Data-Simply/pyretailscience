"""This module provides functionality for performing operations on geographic data using Ibis expressions.

It allows efficient computation of geospatial transformations and analyses on structured data tables.

### Core Features

- **Ibis Integration**: Converts Pandas DataFrames into Ibis tables for optimized query execution.
- **Dynamic Column Selection**: Allows passing latitude, longitude, and other columns as parameters.
- **Haversine Distance Calculation**: Computes great-circle distances between coordinate pairs.
- **Supports Large Datasets**: Leverages Ibis for scalable computation across different backends.
- **Unit Customization**: Defaults to kilometers but allows custom radius values.

### Use Cases

- **Distance-Based Filtering**: Identify locations within a specified radius.
- **Geospatial Analysis**: Examine movement patterns and spatial distributions.
- **Logistics & Routing**: Calculate optimal routes and service areas.

### Limitations and Warnings

- **Assumes Spherical Earth**: The function approximates Earth as a perfect sphere, which may introduce slight inaccuracies.
- **Backend Support Varies**: Ensure the Ibis backend supports trigonometric functions before usage.
"""
import ibis
import pandas as pd


def haversine_distance(
    df: pd.DataFrame | ibis.Table,
    lat_col: str,
    lon_col: str,
    target_lat_col: str,
    target_lon_col: str,
    radius: float = 6371.0,
) -> pd.DataFrame:
    """Converts a Pandas DataFrame into an Ibis table and computes Haversine distances dynamically.

    Parameters:
        df (pd.DataFrame | ibis.Table): The input DataFrame or ibis Table with latitude and longitude columns.
        lat_col (str): Column name for the source latitude.
        lon_col (str): Column name for the source longitude.
        target_lat_col (str): Column name for the target latitude.
        target_lon_col (str): Column name for the target longitude.
        radius (float, optional): Earth's radius in kilometers (default: 6371 km).

    Returns:
        pd.DataFrame: DataFrame with an additional column for computed distances.
    """
    if isinstance(df, pd.DataFrame):
        df: ibis.Table = ibis.memtable(df)

    lat1, lon1, lat2, lon2 = df[lat_col], df[lon_col], df[target_lat_col], df[target_lon_col]

    lat1_rad = lat1.radians()
    lat2_rad = lat2.radians()
    delta_lat = (lat2 - lat1).radians()
    delta_lon = (lon2 - lon1).radians()

    a = (delta_lat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2).sin().pow(2)
    c = 2 * a.sqrt().asin()
    distance = radius * c

    t_with_distance = df.mutate(distance=distance)

    return t_with_distance.execute()
