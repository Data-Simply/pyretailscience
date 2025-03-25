"""This module provides the `HMLSegmentation` class for categorizing customers into spend-based segments.

HMLSegmentation extends `ThresholdSegmentation` and classifies customers into Heavy, Medium, Light,
and optionally Zero spenders based on the Pareto principle (80/20 rule). It is commonly used in retail
to analyze customer spending behavior and optimize marketing strategies.
"""

from typing import Literal

import ibis
import pandas as pd

from pyretailscience.segmentation.threshold import ThresholdSegmentation


class HMLSegmentation(ThresholdSegmentation):
    """Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
    ) -> None:
        """Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend.

        HMLSegmentation is a subclass of ThresholdSegmentation and based around an industry standard definition. The
        thresholds for Heavy (top 20%), Medium (next 30%) and Light (bottom 50%) are chosen based on the pareto
        distribution, commonly know as the 80/20 rule. It is typically used in retail to segment customers based on
        their spend, transaction volume or quantities purchased.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must contain a customer_id column.
            value_col (str, optional): The column to use for the segmentation. Defaults to get_option("column.unit_spend").
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional): How to handle
                customers with zero spend. Defaults to "separate_segment".
        """
        thresholds = [0.500, 0.800, 1]
        segments = ["Light", "Medium", "Heavy"]
        super().__init__(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
        )
