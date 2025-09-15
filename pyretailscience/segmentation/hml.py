"""Heavy-Medium-Light (HML) Segmentation for Customer Value Classification.

## Business Context

The 80/20 rule dominates retail customer behavior: typically 20% of customers generate 80%
of revenue. HML segmentation formalizes this insight by classifying customers into Heavy,
Medium, Light, and Zero spenders, enabling targeted strategies for each value tier.

## The Business Problem

All customers are not equal, but many retailers treat them the same way. Marketing budgets
are wasted on low-value customers while high-value customers don't receive appropriate
attention. Without clear customer value classification, businesses struggle to:
- Allocate marketing spend effectively
- Design appropriate service levels
- Create relevant offers for different customer types
- Identify at-risk high-value customers

## Real-World Applications

### Heavy Spenders (Top ~20%)
- VIP programs with exclusive access and premium support
- Personalized shopping experiences and dedicated account management
- Early access to new products and sales
- Higher-value promotional offers and loyalty rewards

### Medium Spenders (Middle ~30%)
- Growth-focused marketing to move them toward Heavy tier
- Category expansion offers to increase wallet share
- Loyalty programs designed to increase purchase frequency
- Targeted promotions based on purchase history

### Light Spenders (Lower ~50%)
- Cost-effective digital marketing channels
- Basic loyalty programs and promotional offers
- Automated email campaigns for reactivation
- Focus on retention rather than acquisition costs

### Zero Spenders
- Win-back campaigns for previously active customers
- Low-cost reactivation offers
- Analysis for churn prevention insights
- Potential customer file purging for database hygiene

This module extends ThresholdSegmentation to implement the standard HML classification
using Pareto-based percentile thresholds for consistent, business-relevant segments.
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
