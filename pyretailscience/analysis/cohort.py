"""Cohort Analysis and User Segmentation.

This module implements functionality for performing cohort analysis, a powerful technique used in customer analytics
and retention strategies.

Cohort analysis helps in understanding customer behavior over time by grouping users based on shared characteristics
or experiences, such as sign-up date, first purchase, or marketing campaign interaction. This method provides
valuable insights into user engagement, retention, and lifetime value, which businesses can leverage in various ways:

1. Customer retention analysis: By tracking how different cohorts behave over time, businesses can identify trends
   in user engagement and develop strategies to improve customer loyalty.

2. Marketing performance evaluation: Understanding how different user groups respond to marketing efforts helps in
   optimizing campaigns for higher conversions and better ROI.

3. Product lifecycle insights: Analyzing user activity across cohorts can reveal product adoption trends and inform
   feature development or enhancements.

4. Revenue forecasting: Cohort-based revenue tracking enables more accurate predictions of future earnings and
   helps in financial planning.

5. Personalization and segmentation: Businesses can tailor their offerings based on cohort behavior to enhance
   customer experience and increase retention rates.

The module employs key metrics such as retention rate, churn rate, and customer lifetime value (CLV) to measure
cohort performance and user engagement over time:

- Retention Rate: The percentage of users who continue to engage with a product or service over a given period.
- Churn Rate: The percentage of users who stop engaging with the product within a specific timeframe.
- Customer Lifetime Value (CLV): The predicted total revenue a customer will generate throughout their relationship
  with the business.

By leveraging cohort analysis, businesses can make data-driven decisions to enhance customer experience, improve
marketing strategies, and drive long-term growth.
"""

from datetime import date

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import get_option
from pyretailscience.style.tailwind import get_listed_cmap


class CohortPlot:
    """Class for performing cohort analysis and visualization."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        customer_col: str = get_option("column.customer_id"),
        date_col: str = get_option("column.transaction_date"),
        aggregation_func: str = "nunique",
        x_period: str = "month",
        y_period: str = "month",
        start_date: date | None = None,
        end_date: date | None = None,
        percentage: bool = False,
    ) -> None:
        """Initializes the CohortPlot object.

        Args:
            df (pd.DataFrame | ibis.Table): The dataset containing transaction data.
            customer_col (str, optional): Column name representing customer IDs. Defaults
                to option column.customer_id.
            date_col (str, optional): Column name representing transaction dates. Defaults
                to option column.transaction_date.
            aggregation_func (str, optional): The aggregation function to apply (e.g., "nunique", "sum", "mean"). Defaults
                to option nunique.
            x_period (str): Period for cohort acquisition (e.g., "month").
            y_period (str): Period for cohort retention (e.g., "month").
            start_date (Optional[date]): Start date filter for transactions.
            end_date (Optional[date]): End date filter for transactions.
            percentage (bool): If True, converts cohort values into retention percentages relative to the first period.

        Raises:
            ValueError: If `x_period` is not equal to `y_period`.
            ValueError: If `df` is missing required columns (`customer_col` or `date_col`).
        """
        if x_period != y_period:
            error_message = f"x_period ('{x_period}') must be equal to y_period ('{y_period}')."
            raise ValueError(error_message)

        required_cols = [customer_col, date_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_message = f"Missing required columns: {missing_cols}"
            raise ValueError(error_message)

        self.table = self._calculate_cohorts(
            df=df,
            customer_col=customer_col,
            date_col=date_col,
            aggregation_func=aggregation_func,
            x_period=x_period,
            y_period=y_period,
            start_date=start_date,
            end_date=end_date,
            percentage=percentage,
        )

    def _calculate_cohorts(
        self,
        df: pd.DataFrame | ibis.Table,
        customer_col: str = get_option("column.customer_id"),
        date_col: str = get_option("column.transaction_date"),
        aggregation_func: str = "nunique",
        x_period: str = "month",
        y_period: str = "month",
        start_date: date | None = None,
        end_date: date | None = None,
        percentage: bool = False,
    ) -> pd.DataFrame:
        """Computes a cohort analysis table based on transaction data.

        Args:
            df (pd.DataFrame | ibis.Table): The dataset containing transaction data.
            customer_col (str, optional): Column name representing customer IDs. Defaults
                to option column.customer_id.
            date_col (str, optional): Column name representing transaction dates. Defaults
                to option column.transaction_date.
            aggregation_func (str, optional): The aggregation function to apply (e.g., "nunique", "sum", "mean"). Defaults
                to option nunique.
            x_period (str): Period for cohort acquisition (e.g., "month").
            y_period (str): Period for cohort retention (e.g., "month").
            start_date (Optional[date]): Start date filter for transactions.
            end_date (Optional[date]): End date filter for transactions.
            percentage (bool): If True, converts cohort values into retention percentages relative to the first period.

        Returns:
            pd.DataFrame: Cohort analysis table.

        Raises:
            ValueError: start_date must be specified.
        """
        ibis_table = ibis.memtable(df) if isinstance(df, pd.DataFrame) else df
        if start_date is None:
            raise ValueError("start_date must be specified.")
        if end_date is None:
            end_date = ibis_table[date_col].max()

        filtered_table = ibis_table.filter((ibis_table[date_col] >= start_date) & (ibis_table[date_col] <= end_date))
        filtered_table = filtered_table.mutate(
            period_shopped=filtered_table[date_col].truncate(x_period),
            period_value=filtered_table[customer_col],
        )

        customer_cohort = filtered_table.group_by(customer_col).aggregate(
            min_period_shopped=filtered_table.period_shopped.min(),
        )

        cohort_table = (
            filtered_table.join(customer_cohort, [customer_col])
            .group_by("min_period_shopped", "period_shopped")
            .aggregate(period_value=getattr(filtered_table.period_value, aggregation_func)())
        )

        cohort_table = cohort_table.mutate(
            period_since=cohort_table.period_shopped.delta(cohort_table.min_period_shopped, unit=y_period),
        )

        cohort_df = cohort_table.execute().drop_duplicates(subset=["min_period_shopped", "period_since"])

        cohort_analysis_table = cohort_df.pivot(
            index="min_period_shopped",
            columns="period_since",
            values="period_value",
        )

        if percentage:
            cohort_analysis_table = cohort_analysis_table.div(cohort_analysis_table.iloc[:, 0], axis=0).round(2)

        return cohort_analysis_table.fillna(0)

    def plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        group_col: str,
        value_col: str,
        x_label: str | None = None,
        y_label: str | None = None,
        title: str | None = None,
        ax: Axes | None = None,
        source_text: str | None = None,
        cbarlabel: str = "Revenue",
        **kwargs: dict,
    ) -> SubplotBase:
        """Plots a cohort map for the given DataFrame.

        Args:
            df (pd.DataFrame): Dataframe containing cohort analysis data.
            x_col (str): Column name for x-axis labels.
            group_col (str): Column name for y-axis labels.
            value_col (str): Column representing cohort values.
            x_label (str, optional): Label for x-axis.
            y_label (str, optional): Label for y-axis.
            title (str, optional): Title of the plot.
            ax (Axes, optional): Matplotlib axes object to plot on.
            source_text (str, optional): Additional source text annotation.
            cbarlabel (str, optional): Label for the colorbar. Defaults to "Revenue".
            **kwargs: Additional keyword arguments for cohort styling.

        Returns:
            SubplotBase: The matplotlib axes object.
        """
        ax = ax or plt.gca()
        df = df.pivot(index=group_col, columns=x_col, values=value_col)
        cmap = get_listed_cmap("green")
        im = ax.imshow(df, cmap=cmap, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize="x-large")

        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_xticklabels(df.columns, rotation=-30, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(df.index)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.set_xticks(np.arange(df.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(df.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        threshold = im.norm(df.to_numpy().max()) / 2.0
        valfmt = ticker.StrMethodFormatter("{x:,.0f}")
        textcolors = ("black", "white")

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                color = textcolors[int(im.norm(df.iloc[i, j]) > threshold)]
                ax.text(j, i, valfmt(df.iloc[i, j], None), ha="center", va="center", color=color)

        ax = gu.standard_graph_styles(
            ax=ax,
            title=title,
            x_label=x_label or x_col,
            y_label=y_label or group_col,
        )
        ax.grid(False)
        ax.hlines(y=3 - 0.5, xmin=-0.5, xmax=df.shape[1] - 0.5, color="white", linewidth=4)

        if source_text:
            gu.add_source_text(ax=ax, source_text=source_text, is_venn_diagram=True)

        return gu.standard_tick_styles(ax)
