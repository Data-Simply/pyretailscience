import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from pyretailscience.data.contracts import TransactionItemLevelContract
from pyretailscience.style.graph_utils import human_format, standard_graph_styles
from pyretailscience.style.tailwind import COLORS
import operator


class PurchasesPerCustomer:
    """A class to plot the distribution of the number of purchases per customer.

    Attributes:
        cust_purchases_s (pd.Series): The number of purchases per customer.

    Args:
        df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
            TransactionItemLevelContract.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if TransactionItemLevelContract(df).validate() is False:
            raise ValueError("The dataframe does not comply with the TransactionItemLevelContract")

        self.cust_purchases_s = df.groupby("customer_id").size()

    def plot(
        self,
        bins: int = 10,
        cumlative: bool = False,
        ax: Axes | None = None,
        draw_percentile_line: bool = False,
        percentile_line: float = 0.5,
        source_text: str = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the distribution of the number of purchases per customer.

        Args:
            bins (int, optional): The number of bins to plot. Defaults to 10.
            cumlative (bool, optional): Whether to plot the cumulative distribution. Defaults to False.
            ax (Axes, optional): The Matplotlib axes to plot the graph on. Defaults to None.
            draw_percentile_line (bool, optional): Whether to draw a line at the percentile specified with
                the `percentile_line` paramter. Defaults to False.
            percentile_line (float, optional): The percentile to draw a line at. Defaults to 0.8.

        Returns:
            SubplotBase: The Matplotlib axes of the plot
        """

        density = False
        if cumlative:
            density = True

        ax = self.cust_purchases_s.hist(
            bins=bins,
            cumulative=cumlative,
            ax=ax,
            density=density,
            color=COLORS["green"][500],
            **kwargs,
        )
        ax.set_xlabel("Number of purchases")
        ax.xaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        ax = standard_graph_styles(ax)

        if cumlative:
            plt.title("Number of Purchases Cumulative Distribution")
            plt.ylabel("Percentage of customers")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        else:
            plt.title("Number of Purchases Distribution")
            plt.ylabel("Number of customers")
            ax.yaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        if draw_percentile_line:
            if percentile_line > 1 or percentile_line < 0:
                raise ValueError("Percentile line must be between 0 and 1")
            ax.axvline(
                x=self.purchases_percentile(percentile_line),
                color=COLORS["red"][500],
                linestyle="--",
                lw=2,
                label=f"{percentile_line:.1%} of customers",
            )
            ax.legend(frameon=False)

        if source_text:
            ax.annotate(
                source_text,
                xy=(-0.1, -0.2),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=10,
            )

        return ax

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Get the number of purchases at a given percentile.

        Args:
            percentile (float): The percentile to get the number of purchases at.

        Returns:
            float: The number of purchases at the given percentile.
        """
        return self.cust_purchases_s.quantile(percentile)

    def find_purchase_percentile(self, number_of_purchases: int, comparison: str = "less_than_equal_to") -> float:
        """Find the percentile of the number of purchases.

        Args:
            number_of_purchases (int): The number of purchases to find the percentile of.

        Returns:
            float: The percentile of the number of purchases.
        """
        ops = {
            "less_than": operator.lt,
            "less_than_equal_to": operator.le,
            "equal_to": operator.eq,
            "not_equal_to": operator.ne,
            "greater_than": operator.gt,
            "greater_than_equal_to": operator.ge,
        }

        if comparison not in ops:
            raise ValueError(
                "Comparison must be one of 'less_than', 'less_than_equal_to', 'equal_to', 'not_equal_to', "
                "'greater_than', 'greater_than_equal_to'"
            )

        return len(self.cust_purchases_s[ops[comparison](self.cust_purchases_s, number_of_purchases)]) / len(
            self.cust_purchases_s
        )


class DaysBetweenPurchases:
    """A class to plot the distribution of the average number of days between purchases per customer.

    Attributes:
        purchase_dist_s (pd.Series): The average number of days between purchases per customer.

    Args:
        df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
            TransactionItemLevelContract.

    Raises:
        ValueError: If the dataframe does not comply with the TransactionItemLevelContract
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if TransactionItemLevelContract(df).validate() is False:
            raise ValueError("The dataframe does not comply with the TransactionItemLevelContract")

        purchase_dist_df = df[["customer_id", "transaction_datetime"]].copy()
        purchase_dist_df["transaction_datetime"] = df["transaction_datetime"].dt.floor("D")
        purchase_dist_df = purchase_dist_df.drop_duplicates().sort_values(["customer_id", "transaction_datetime"])
        purchase_dist_df["diff"] = purchase_dist_df.groupby("customer_id")["transaction_datetime"].transform(
            lambda x: x.diff()
        )
        purchase_dist_df = purchase_dist_df[~purchase_dist_df["diff"].isnull()]
        purchase_dist_df["diff"] = purchase_dist_df["diff"].dt.days
        self.purchase_dist_s = purchase_dist_df.groupby("customer_id")["diff"].mean()

    def plot(
        self,
        bins: int = 10,
        cumlative: bool = False,
        ax: Axes | None = None,
        draw_percentile_line: bool = False,
        percentile_line: float = 0.5,
        source_text: str = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the distribution of the average number of days between purchases per customer.

        Args:
            bins (int, optional): The number of bins to plot. Defaults to 10.
            cumlative (bool, optional): Whether to plot the cumulative distribution. Defaults to False.
            ax (Axes, optional): The Matplotlib axes to plot the graph on. Defaults to None.
            draw_percentile_line (bool, optional): Whether to draw a line at the percentile specified with
                the `percentile_line` paramter. Defaults to False.
            percentile_line (float, optional): The percentile to draw a line at. Defaults to 0.8.

        Returns:
            SubplotBase: The Matplotlib axes of the plot
        """
        density = False
        if cumlative:
            density = True

        ax = self.purchase_dist_s.hist(
            bins=bins,
            cumulative=cumlative,
            ax=ax,
            density=density,
            color=COLORS["green"][500],
            **kwargs,
        )
        plt.xlabel("Average Number of Days Between Purchases")
        ax.xaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        ax = standard_graph_styles(ax)

        if cumlative:
            plt.title("Average Days Between Purchases Cumulative Distribution")
            plt.ylabel("Percentage of Customers")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        else:
            plt.title("Average Days Between Purchases Distribution")
            plt.ylabel("Number of Customers")
            ax.yaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        if draw_percentile_line:
            if percentile_line > 1 or percentile_line < 0:
                raise ValueError("Percentile line must be between 0 and 1")
            ax.axvline(
                x=self.purchases_percentile(percentile_line),
                color=COLORS["red"][500],
                linestyle="--",
                lw=2,
                label=f"{percentile_line:.1%} of customers",
                ymax=0.96,
            )
            ax.legend(frameon=False)

        if source_text:
            ax.annotate(
                source_text,
                xy=(-0.1, -0.2),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=10,
            )

        return ax

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Get the average number of days between purchases at a given percentile.

        Args:
            percentile (float): The percentile to get the average number of days between purchases at.

        Returns:
            float: The average number of days between purchases at the given percentile.
        """
        return self.purchase_dist_s.quantile(percentile)


class TransactionChurn:
    """A class to plot the churn rate by number of purchases.

    Attributes:
        purchase_dist_df (pd.DataFrame): The churn rate by number of purchases.
        n_unique_customers (int): The number of unique customers in the dataframe.

    Args:
        df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
            TransactionItemLevelContract.
        churn_period (float): The number of days to consider a customer churned.
    """

    def __init__(self, df: pd.DataFrame, churn_period: float) -> None:
        if TransactionItemLevelContract(df).validate() is False:
            raise ValueError("The dataframe does not comply with the TransactionItemLevelContract")

        purchase_dist_df = df[["customer_id", "transaction_datetime"]].copy()
        # Truncate the transaction_datetime to the day
        purchase_dist_df["transaction_datetime"] = df["transaction_datetime"].dt.floor("D")
        purchase_dist_df = purchase_dist_df.drop_duplicates()
        purchase_dist_df = purchase_dist_df.sort_values(["customer_id", "transaction_datetime"])
        purchase_dist_df["transaction_number"] = purchase_dist_df.groupby("customer_id").cumcount() + 1

        purchase_dist_df["last_transaction"] = (
            purchase_dist_df.groupby("customer_id")["transaction_datetime"].shift(-1).isna()
        )
        purchase_dist_df["transaction_before_churn_window"] = purchase_dist_df["transaction_datetime"] < (
            purchase_dist_df["transaction_datetime"].max() - pd.Timedelta(days=churn_period)
        )
        purchase_dist_df["churned"] = (
            purchase_dist_df["last_transaction"] & purchase_dist_df["transaction_before_churn_window"]
        )

        purchase_dist_df = (
            purchase_dist_df[purchase_dist_df["transaction_before_churn_window"]]
            .groupby(["transaction_number"])["churned"]
            .value_counts()
            .unstack()
        )
        purchase_dist_df.columns = ["retained", "churned"]
        purchase_dist_df["churned_pct"] = purchase_dist_df["churned"].div(purchase_dist_df.sum(axis=1))
        self.purchase_dist_df = purchase_dist_df

        self.n_unique_customers = df["customer_id"].nunique()

    def plot(
        self,
        cumlative: bool = False,
        ax: Axes | None = None,
        source_text: str = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the churn rate by number of purchases.

        Args:
            cumlative (bool, optional): Whether to plot the cumulative distribution. Defaults to False.
            ax (Axes, optional): The Matplotlib axes to plot the graph on. Defaults to None.

        Returns:
            SubplotBase: The Matplotlib axes of the plot
        """
        if cumlative:
            cumulative_churn_rate_s = self.purchase_dist_df["churned"].cumsum().div(self.n_unique_customers)
            ax = cumulative_churn_rate_s.plot.area(color=COLORS["green"][500])
            ax.set_xlim(self.purchase_dist_df.index.min(), self.purchase_dist_df.index.max())
        else:
            ax = self.purchase_dist_df["churned_pct"].plot.bar(
                rot=0,
                color=COLORS["green"][500],
                width=0.8,
                **kwargs,
            )

        standard_graph_styles(ax)

        # Format y axis as a percent
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_xlabel("Number of Purchases")
        ax.set_ylabel("% Churned")
        ax.set_title("Churn Rate by Number of Purchases")

        if source_text:
            ax.annotate(
                source_text,
                xy=(-0.1, -0.2),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=10,
            )

        return ax
