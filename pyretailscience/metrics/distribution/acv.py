"""ACV (All Commodity Volume) metric.

ACV measures total dollar sales across all products in a set of stores,
expressed in millions ($MM).
"""

from __future__ import annotations

import ibis
import pandas as pd
from ibis import _

from pyretailscience.options import get_option
from pyretailscience.utils.validation import validate_columns


class Acv:
    """Calculates ACV (All Commodity Volume) for a set of stores.

    ACV represents total dollar sales across all products, expressed in millions ($MM).
    NaN values in the spend column are excluded from the sum.

    Results are accessible via the `table` attribute (ibis Table) or the `df` property
    (materialized pandas DataFrame).

    Args:
        df (pd.DataFrame | ibis.Table): Transaction data containing at least a unit_spend column.
        group_by (str | list[str] | None, optional): Optional column(s) to group the ACV calculation by
            (e.g., store_id). Defaults to None for total ACV.
        acv_scale_factor (float, optional): Factor to scale the ACV result (default is 1,000,000 for $MM).

    Raises:
        TypeError: If df is not a pandas DataFrame or an Ibis Table.
        ValueError: If required columns are missing from the data or if acv_scale_factor is not positive.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        group_by: str | list[str] | None = None,
        acv_scale_factor: float = 1_000_000,
    ) -> None:
        """Initializes the ACV calculation."""
        self._df: pd.DataFrame | None = None
        self.table: ibis.Table

        if acv_scale_factor <= 0:
            raise ValueError("acv_scale_factor must be positive.")

        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)
        elif not isinstance(df, ibis.Table):
            raise TypeError("df must be either a pandas DataFrame or an Ibis Table.")

        unit_spend_col = get_option("column.unit_spend")

        if isinstance(group_by, str):
            group_by = [group_by]

        required_cols = [unit_spend_col]
        if group_by is not None:
            required_cols.extend(group_by)
            validate_columns(df, required_cols)
            df = df.group_by(group_by)
        else:
            validate_columns(df, required_cols)

        self.table = df.aggregate(acv=_[unit_spend_col].sum() / acv_scale_factor)

    @property
    def df(self) -> pd.DataFrame:
        """Returns the materialized pandas DataFrame of ACV results.

        Returns:
            pd.DataFrame: DataFrame with ACV values. Cached after first access.
        """
        if self._df is None:
            self._df = self.table.execute()
        return self._df
