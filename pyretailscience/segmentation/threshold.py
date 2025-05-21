"""Threshold-Based Customer Segmentation Module.

This module provides the `ThresholdSegmentation` class, which segments customers
based on user-defined thresholds and segment mappings.

Key Features:
- Segments customers based on specified percentile thresholds.
- Uses a specified column for segmentation, with an aggregation function applied.
- Handles customers with zero spend using configurable options.
- Utilizes Ibis for efficient query execution.
"""

from typing import Literal

import ibis
import pandas as pd

from pyretailscience.options import get_option
from pyretailscience.segmentation.base import BaseSegmentation


class ThresholdSegmentation(BaseSegmentation):
    """Segments customers based on user-defined thresholds and segments."""

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        thresholds: list[float],
        segments: list[str],
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_segment_name: str = "Zero",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
    ) -> None:
        """Segments customers based on user-defined thresholds and segments.

        Args:
            df (pd.DataFrame | ibis.Table): A dataframe with the transaction data. The dataframe must contain a customer_id column.
            thresholds (List[float]): The percentile thresholds for segmentation.
            segments (List[str]): A list of segment names for each threshold.
            value_col (str, optional): The column to use for the segmentation. Defaults to get_option("column.unit_spend").
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_segment_name (str, optional): The name of the segment for customers with zero spend. Defaults to "Zero".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional): How to handle
                customers with zero spend. Defaults to "separate_segment".

        Raises:
            ValueError: If the dataframe is missing the columns option column.customer_id or `value_col`, or these
                columns contain null values.
        """
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("The thresholds must be unique.")

        if len(thresholds) != len(segments):
            raise ValueError("The number of thresholds must match the number of segments.")

        if isinstance(df, pd.DataFrame):
            df: ibis.Table = ibis.memtable(df)

        value_col = get_option("column.unit_spend") if value_col is None else value_col

        required_cols = [get_option("column.customer_id"), value_col]

        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        df = df.group_by(get_option("column.customer_id")).aggregate(
            **{value_col: getattr(df[value_col], agg_func)()},
        )

        # Separate customers with zero spend
        zero_df = None
        if zero_value_customers == "exclude":
            df = df.filter(df[value_col] != 0)
        elif zero_value_customers == "separate_segment":
            zero_df = df.filter(df[value_col] == 0).mutate(segment_name=ibis.literal(zero_segment_name))
            df = df.filter(df[value_col] != 0)

        window = ibis.window(order_by=ibis.asc(df[value_col]))
        df = df.mutate(ptile=ibis.percent_rank().over(window))

        case_args = [(df["ptile"] <= quantile, segment) for quantile, segment in zip(thresholds, segments, strict=True)]

        df = df.mutate(segment_name=ibis.cases(*case_args)).drop(["ptile"])

        if zero_value_customers == "separate_segment":
            df = ibis.union(df, zero_df)

        self.table = df

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names."""
        if self._df is None:
            self._df = self.table.execute().set_index(get_option("column.customer_id"))
        return self._df
