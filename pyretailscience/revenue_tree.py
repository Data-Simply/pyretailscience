"""Revenue Tree Analysis Module.

This module implements a Revenue Tree analysis for retail businesses. The Revenue Tree
is a hierarchical breakdown of factors contributing to overall revenue, allowing for
detailed analysis of sales performance and identification of areas for improvement.


Key Components of the Revenue Tree:

1. Revenue: The top-level metric, calculated as Customers * Revenue per Customer.

2. Customers: Total number of customers, broken down into:
   - Returning Customers: Existing customers making repeat purchases.
   - New Customers: First-time buyers.

3. Revenue per Customer: Average revenue generated per customer, calculated as:
   Orders per Customer * Average Order Value.

4. Orders per Customer: Average number of orders placed by each customer.

5. Average Order Value: Average monetary value of each order, calculated as:
   Items per Order * Price per Item.

6. Items per Order: Average number of items in each order.

7. Price per Item: Average price of each item sold.

This module can be used to create, update, and analyze Revenue Tree data structures
for retail businesses, helping to identify key drivers of revenue changes and
inform strategic decision-making.
"""

import pandas as pd

from pyretailscience.data.contracts import CustomContract, build_expected_columns, build_non_null_columns


class RevenueTree:
    """Revenue Tree Analysis Class."""

    def __init__(
        self,
        df: pd.DataFrame,
        p1_index: list[bool] | pd.Series,
        p2_index: list[bool] | pd.Series,
        group_col: str | None = None,
        pre_aggregated: bool = False,
    ) -> None:
        """Initialize the Revenue Tree Analysis Class.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.
            p1_index (list[bool] | pd.Series): A boolean index for the first period.
            p2_index (list[bool] | pd.Series): A boolean index for the second period.
            group_col (str, optional): The column to group the data by. Defaults to None.
            pre_aggregated (bool, optional): Whether the data is pre-aggregated. Defaults to False.

        Raises:
            ValueError: If the required columns are not present in the DataFrame.
            ValueError: If the lengths of p1_index, p2_index, and df are not equal.

        Example:
            >>> import pandas as pd
            >>> from pyretailscience import RevenueTree
            >>> data = {
            ...     "customer_id": [1, 2, 3, 4, 5, 6],
            ...     "transaction_id": [1, 2, 3, 4, 5, 6],
            ...     "total_price": [100, 200, 300, 400, 500, 600],
            ...     "quantity": [1, 2, 3, 4, 5, 6],
            ... }
            >>> df = pd.DataFrame(data)
            >>> p1_index = [True, False, True, False, True, False]
            >>> p2_index = [False, True, False, True, False, True]
            >>> rev_tree = RevenueTree(df=df, p1_index=p1_index, p2_index=p2_index)
        """
        if pre_aggregated:
            required_cols = ["customers", "tranactions", "total_price"]
        else:
            required_cols = ["customer_id", "transaction_id", "total_price"]

        if "quantity" in df.columns:
            required_cols.append("quantity")
        if group_col is not None:
            required_cols.append(group_col)

        contract = CustomContract(
            df,
            basic_expectations=build_expected_columns(columns=required_cols),
            extended_expectations=build_non_null_columns(columns=required_cols),
        )
        if contract.validate() is False:
            msg = f"The dataframe requires the columns {required_cols} and they must be non-null"
            raise ValueError(msg)

        if not len(p1_index) == len(p2_index) == len(df):
            raise ValueError("p1_index, p2_index, and df should have the same length")

        if pre_aggregated is False:
            df, p1_index, p2_index = self._agg_data(df=df, p1_index=p1_index, p2_index=p2_index)

        self.revenue_tree_df = self._calc_tree_kpis(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
        )

    @staticmethod
    def _agg_data(
        df: pd.DataFrame,
        p1_index: list[bool] | pd.Series,
        p2_index: list[bool] | pd.Series,
        group_col: str | None = None,
    ) -> tuple[pd.DataFrame, list[bool], list[bool]]:
        if group_col is not None:
            p1_group = df[p1_index].groupby(group_col)
            p2_group = df[p2_index].groupby(group_col)
            p1_df = p1_group.agg(
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                total_price=("total_price", "sum"),
            )
            p2_df = p2_group.agg(
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                total_price=("total_price", "sum"),
            )
            if "quantity" in df.columns:
                p1_df["quantity"] = p1_group["quantity"].sum()
                p2_df["quantity"] = p2_group["quantity"].sum()
        else:
            p1_df = df[p1_index]
            p2_df = df[p2_index]
            p1_df = pd.DataFrame(
                {
                    "customers": p1_df["customer_id"].nunique(),
                    "transactions": p1_df["transaction_id"].nunique(),
                    "total_price": p1_df["total_price"].sum(),
                },
                index=["p1"],
            )
            p2_df = pd.DataFrame(
                {
                    "customers": p2_df["customer_id"].nunique(),
                    "transactions": p2_df["transaction_id"].nunique(),
                    "total_price": p2_df["total_price"].sum(),
                },
                index=["p2"],
            )
            if "quantity" in df.columns:
                p1_df["quantity"] = df[p1_index]["quantity"].sum()
                p2_df["quantity"] = df[p2_index]["quantity"].sum()

        new_p1_index = [True] * len(p1_df) + [False] * len(p2_df)
        new_p2_index = [not i for i in new_p1_index]

        return pd.concat([p1_df, p2_df]), new_p1_index, new_p2_index

    @staticmethod
    def _calc_tree_kpis(
        df: pd.DataFrame,
        p1_index: list[bool] | pd.Series,
        p2_index: list[bool] | pd.Series,
    ) -> pd.DataFrame:
        df["total_price_per_cust"] = df["total_price"] / df["customers"]
        df["total_price_per_transaction"] = df["total_price"] / df["transactions"]
        df["frequency"] = df["transactions"] / df["customers"]

        p1_df = df[p1_index]
        p1_df.columns = [f"{col}_p1" for col in p1_df.columns]
        p2_df = df[p2_index]
        p2_df.columns = [f"{col}_p2" for col in p2_df.columns]

        if set(df.index.to_list()) == {"p1", "p2"}:
            p1_df = p1_df.reset_index(drop=True)
            p2_df = p2_df.reset_index(drop=True)

        df = pd.concat([p1_df, p2_df], axis=1)

        # Calculations
        df["customers_diff"] = df["customers_p2"] - df["customers_p1"]
        df["transactions_diff"] = df["transactions_p2"] - df["transactions_p1"]
        df["total_price_diff"] = df["total_price_p2"] - df["total_price_p1"]
        df["total_price_per_cust_diff"] = df["total_price_per_cust_p2"] - df["total_price_per_cust_p1"]
        df["total_price_per_transaction_diff"] = (
            df["total_price_per_transaction_p2"] - df["total_price_per_transaction_p1"]
        )
        df["frequency_diff"] = df["frequency_p2"] - df["frequency_p1"]

        df["customers_pc"] = df["customers_diff"] / df["customers_p1"]
        df["transactions_pc"] = df["transactions_diff"] / df["transactions_p1"]
        df["total_price_pc"] = df["total_price_diff"] / df["total_price_p1"]
        df["total_price_per_cust_pc"] = df["total_price_per_cust_diff"] / df["total_price_per_cust_p1"]
        df["total_price_per_transaction_pc"] = (
            df["total_price_per_transaction_diff"] / df["total_price_per_transaction_p1"]
        )
        df["frequency_pc"] = df["frequency_diff"] / df["frequency_p1"]

        df["customers_contrib"] = (
            df["total_price_p2"]
            - (df["customers_p1"] * df["total_price_per_cust_p2"])
            - ((df["customers_diff"] * df["total_price_per_cust_diff"]) / 2)
        )
        df["total_price_per_cust_contrib"] = (
            df["total_price_p2"]
            - (df["total_price_per_cust_p1"] * df["customers_p2"])
            - ((df["customers_diff"] * df["total_price_per_cust_diff"]) / 2)
        )

        df["frequency_contrib"] = (
            (
                df["total_price_per_cust_p2"]
                - (df["frequency_p1"] * df["total_price_per_transaction_p2"])
                - ((df["frequency_diff"] * df["total_price_per_transaction_diff"]) / 2)
            )
            * df["customers_p2"]
        ) - ((df["customers_diff"] * df["total_price_per_cust_diff"]) / 4)
        df["total_price_per_transaction_contrib"] = (
            (
                df["total_price_per_cust_p2"]
                - (df["total_price_per_transaction_p1"] * df["frequency_p2"])
                - ((df["frequency_diff"] * df["total_price_per_transaction_diff"]) / 2)
            )
            * df["customers_p2"]
        ) - ((df["customers_diff"] * df["total_price_per_cust_diff"]) / 4)

        return df
