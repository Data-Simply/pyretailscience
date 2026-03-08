"""Shared validation utilities for PyRetailScience."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ibis
    import pandas as pd


def validate_columns(df: pd.DataFrame | ibis.Table, required_cols: list[str]) -> None:
    """Validates that a DataFrame or Ibis Table contains all required columns.

    Args:
        df (pd.DataFrame | ibis.Table): The data to validate.
        required_cols (list[str]): Column names that must be present.

    Raises:
        ValueError: If any required columns are missing from the data.
    """
    if isinstance(required_cols, str):
        msg = "required_cols must be a list of column names, not a string."
        raise TypeError(msg)
    missing_cols = sorted(set(required_cols) - set(df.columns))
    if len(missing_cols) > 0:
        msg = f"The following columns are required but missing: {missing_cols}"
        raise ValueError(msg)
