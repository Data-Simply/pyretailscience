"""Contribution Analysis module for retail metrics.

This module provides a ContributionAnalysis class to perform contribution analysis
between two time periods, calculating expected vs. actual differences for key business metrics.
"""

import ibis
import pandas as pd

from pyretailscience.revenue_tree import RevenueTree


class ContributionAnalysis:
    """Class for performing contribution analysis comparing two time periods.

    This analysis identifies how different segments (product/store combinations)
    are performing compared to expectations based on overall trends.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        group_cols: list[str],
        p1_label: str,
        p2_label: str,
        period_col: str,
    ) -> pd.DataFrame:
        """Performs contribution analysis comparing two time periods.

        Args:
            df: DataFrame or Ibis Table containing transaction data
            group_cols (list[str]): Columns to group by for the analysis
            p1_label: Custom label for period 1
            p2_label: Custom label for period 2
            period_col: The column of the table containing the periods

        Returns:
            DataFrame containing contribution analysis results
        """
        # TODO: Replace these with the option values
        self.base_metrics = [
            "spend",
            "transactions",
            "customers",
            "units",
            "price_per_unit",
            "transactions_per_customer",
            "units_per_transaction",
            "spend_per_transaction",
            "spend_per_customer",
        ]
        self.raw_data = self._perform_aggregation(df, period_col, group_cols)

        self.df = self._process_data(p1_label, p2_label, group_cols)

    def _perform_aggregation(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        group_cols: list[str],
    ) -> str:
        """Generates SQL query for contribution analysis.

        Args:
            df (pd.DataFrame | ibis.Table): DataFrame or Ibis Table containing transaction data
            period_col (str): The column of the table containing the periods
            group_cols (list[str]): Columns to group by for the analysis

        Returns:
            SQL query string
        """
        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df, name="my_table")

        group_cols_sql = ", ".join(group_cols)
        sql = f"""
            WITH trans as (
                SELECT  {period_col} as period,  -- label for period
                        {group_cols_sql},
                        transaction_id,
                        customer_id,
                        CAST(SUM(unit_quantity) as FLOAT) AS unit_quantity,
                        CAST(SUM(unit_spend) as INT) AS unit_spend
                   FROM my_table
                  GROUP BY {period_col} ,
                           {group_cols_sql},
                           transaction_id,
                           customer_id
                        )
                 SELECT i.period,
                        {group_cols_sql},
                        spend,
                        transactions,
                        customers,
                        units,
                        total_spend - spend as not_spend,
                        total_transactions - transactions as not_transactions,
                        total_customers - customers as not_customers,
                        total_units - units as not_units
                FROM (
                 SELECT period,
                        {group_cols_sql},
                        CAST(SUM(unit_spend) as FLOAT) as spend,
                        COUNT(DISTINCT transaction_id) as transactions,
                        COUNT(DISTINCT customer_id) as customers,
                        SUM(unit_quantity) as units
                    FROM trans
                GROUP BY period,
                        {group_cols_sql}
                     ) i
              CROSS JOIN (
                          SELECT period,
                                 SUM(unit_spend) as total_spend,
                                 COUNT(DISTINCT transaction_id) as total_transactions,
                                 COUNT(DISTINCT customer_id) as total_customers,
                                 SUM(unit_quantity) as total_units
                            FROM trans
                        GROUP BY period
                        ) s
                WHERE i.period = s.period
                ORDER BY 1,2
        """  # noqa: S608
        return df.sql(sql).to_pandas()

    def _process_data(
        self,
        p1_label: str,
        p2_label: str,
        group_cols: list[str],
    ) -> pd.DataFrame:
        """Processes the raw SQL results into contribution analysis metrics.

        Returns:
            DataFrame with contribution analysis results

        Args:
            p1_label (str): Custom label for period 1
            p2_label (str): Custom label for period 2
            group_cols (list[str]): Columns to group by for the analysis
        """
        if self.raw_data is None:
            raise ValueError("No data available. Run the aggregation first.")

        df = self.raw_data

        # Process "not" metrics
        not_df = df[[*["period", "not_spend", "not_transactions", "not_customers", "not_units"], *group_cols]].copy()
        not_df.columns = [col.replace("not_", "") if "not_" in col else col for col in not_df.columns]

        # Calculate revenue tree metrics for "not" data
        not_df = not_df.set_index(group_cols)
        not_rt_df = RevenueTree._calc_tree_kpis(
            df=not_df,
            p1_index=not_df["period"] == p1_label,
            p2_index=not_df["period"] == p2_label,
        )

        # Calculate revenue tree metrics for actual data
        df = df.set_index(group_cols)
        rt_df = RevenueTree._calc_tree_kpis(
            df=df,
            p1_index=df["period"] == p1_label,
            p2_index=df["period"] == p2_label,
        )
        # Merge actual and "not" metrics
        expected_df = rt_df.merge(not_rt_df, left_index=True, right_index=True, suffixes=("", "_not"))

        # Calculate expected values for period 2
        for metric in self.base_metrics:
            expected_df[f"expected_{metric}_p2"] = expected_df[f"{metric}_p1"] * (
                1 + expected_df[f"{metric}_pct_diff_not"]
            )

        # Calculate expected vs actual differences
        for metric in self.base_metrics:
            expected_df[f"expected_vs_actual_{metric}_diff"] = (
                expected_df[f"{metric}_p2"] - expected_df[f"expected_{metric}_p2"]
            )

        return expected_df.reset_index()
