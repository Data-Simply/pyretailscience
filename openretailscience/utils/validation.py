"""Shared validation utilities for OpenRetailScience."""

from __future__ import annotations

import ibis
import pandas as pd


def ensure_ibis_table(df: pd.DataFrame | ibis.Table) -> ibis.Table:
    """Convert pandas DataFrame to ibis Table, or validate input is an ibis Table.

    Args:
        df (pd.DataFrame | ibis.Table): Input data to convert or validate.

    Returns:
        ibis.Table: An ibis Table representation of the input data.

    Raises:
        TypeError: If df is neither a pandas DataFrame nor an ibis Table.
    """
    if isinstance(df, pd.DataFrame):
        return ibis.memtable(df)
    if isinstance(df, ibis.Table):
        return df
    raise TypeError("df must be either a pandas DataFrame or an Ibis Table.")


def validate_columns(df: pd.DataFrame | ibis.Table, required_cols: list[str]) -> None:
    """Validates that a DataFrame or Ibis Table contains all required columns.

    Args:
        df (pd.DataFrame | ibis.Table): The data to validate.
        required_cols (list[str]): Column names that must be present.

    Raises:
        TypeError: If required_cols is not a list.
        ValueError: If any required columns are missing from the data.
    """
    if not isinstance(required_cols, list):
        msg = "required_cols must be a list of column names."
        raise TypeError(msg)
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if len(missing_cols) > 0:
        msg = f"The following columns are required but missing: {missing_cols}"
        raise ValueError(msg)
