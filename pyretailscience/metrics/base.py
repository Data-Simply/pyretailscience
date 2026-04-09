"""Shared ibis expression helpers for metric calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import pandas as pd

if TYPE_CHECKING:
    import ibis.expr.types as ir

PERCENTAGE_SCALE = 100


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


def ratio_metric(
    numerator: ir.NumericValue,
    denominator: ir.NumericValue,
    scale: float = PERCENTAGE_SCALE,
) -> ir.FloatingValue:
    """Computes a scaled ratio, returning NULL on zero denominator.

    Args:
        numerator (ir.NumericValue): The numerator ibis expression.
        denominator (ir.NumericValue): The denominator ibis expression.
        scale (float, optional): Multiplicative scale factor. Defaults to 100 for percentages.

    Returns:
        ir.FloatingValue: The scaled ratio expression. Evaluates to NULL
            when the denominator is zero (materializes as NaN in pandas).
    """
    return (numerator / denominator.nullif(0)) * scale
