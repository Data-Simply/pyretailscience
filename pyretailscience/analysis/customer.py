"""Classes and function to assist with customer retention analysis."""

import operator

import matplotlib.ticker as mtick
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import pyretailscience.style.graph_utils as gu
from pyretailscience.options import ColumnHelper
from pyretailscience.style.graph_utils import human_format, standard_graph_styles
from pyretailscience.style.tailwind import COLORS


class PurchasesPerCustomer:
    """A class to plot the distribution of the number of purchases per customer.

    Attributes:
        cust_purchases_s (pd.Series): The number of purchases per customer.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the PurchasesPerCustomer class.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
                contain customer_id and transaction_id columns, which must be non-null.

        Raises:
            ValueError: If the dataframe doesn't contain the columns customer_id and transaction_id, or if the columns
                are null.

        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_id]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.cust_purchases_s = df.groupby(cols.customer_id)[cols.transaction_id].nunique()

    def plot(
        self,
        bins: int = 10,
        cumulative: bool = False,
        ax: Axes | None = None,
        percentile_line: float | None = None,
        source_text: str | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the distribution of the number of purchases per customer.

        Args:
            bins (int, optional): The number of bins to plot. Defaults to 10.
            cumulative (bool, optional): Whether to plot the cumulative distribution. Defaults to False.
            ax (Axes, optional): The Matplotlib axes to plot the graph on. Defaults to None.
            percentile_line (float, optional): The percentile to draw a line at. Defaults to None. When None then no
                line is drawn.
            source_text (str, optional): The source text to add to the plot. Defaults to None.
            title (str, optional): The title of the plot. Defaults to None.
            x_label (str, optional): The x-axis label. Defaults to None.
            y_label (str, optional): The y-axis label. Defaults to None.
            kwargs (dict[str, any]): Additional keyword arguments to pass to the plot function.

        Returns:
            SubplotBase: The Matplotlib axes of the plot
        """
        density = False
        if cumulative:
            density = True

        if x_label is None:
            x_label = "Number of purchases"

        ax = self.cust_purchases_s.hist(
            bins=bins,
            cumulative=cumulative,
            ax=ax,
            density=density,
            color=COLORS["green"][500],
            **kwargs,
        )

        ax.xaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        if cumulative:
            default_title = "Number of Purchases cumulative Distribution"
            default_y_label = "Percentage of customers"
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        else:
            default_title = "Number of Purchases Distribution"
            default_y_label = "Number of customers"
            ax.yaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        ax = standard_graph_styles(
            ax,
            title=gu.not_none(title, default_title),
            x_label=x_label,
            y_label=gu.not_none(y_label, default_y_label),
        )

        if percentile_line is not None:
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
            gu.add_source_text(ax=ax, source_text=source_text)

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
            comparison (str, optional): The comparison to use. Defaults to "less_than_equal_to". Must be one of
                less_than, less_than_equal_to, equal_to, not_equal_to, greater_than, or greater_than_equal_to.

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
                "Comparison must be one of 'less_than', 'less_than_equal_to', 'equal_to', 'not_equal_to',",
                "'greater_than', 'greater_than_equal_to'",
            )

        return len(self.cust_purchases_s[ops[comparison](self.cust_purchases_s, number_of_purchases)]) / len(
            self.cust_purchases_s,
        )


class DaysBetweenPurchases:
    """A class to plot the distribution of the average number of days between purchases per customer.

    Attributes:
        purchase_dist_s (pd.Series): The average number of days between purchases per customer.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the DaysBetweenPurchases class.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must have the columns customer_id
                and transaction_date, which must be non-null.

        Raises:
            ValueError: If the dataframe does doesn't contain the columns customer_id and transaction_id, or if the
                columns are null.

        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_date]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.purchase_dist_s = self._calculate_days_between_purchases(df)

    @staticmethod
    def _calculate_days_between_purchases(df: pd.DataFrame) -> pd.Series:
        """Calculate the average number of days between purchases per customer.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must have the columns customer_id
                and transaction_date, which must be non-null.

        Returns:
            pd.Series: The average number of days between purchases per customer.
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_date]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        purchase_dist_df = df[[cols.customer_id, cols.transaction_date]].copy()
        purchase_dist_df[cols.transaction_date] = df[cols.transaction_date].dt.floor("D")
        purchase_dist_df = purchase_dist_df.drop_duplicates().sort_values([cols.customer_id, cols.transaction_date])
        purchase_dist_df["diff"] = purchase_dist_df[cols.transaction_date].diff()
        new_cust_mask = purchase_dist_df[cols.customer_id] != purchase_dist_df[cols.customer_id].shift(1)
        purchase_dist_df = purchase_dist_df[~new_cust_mask]
        purchase_dist_df["diff"] = purchase_dist_df["diff"].dt.days
        return purchase_dist_df.groupby(cols.customer_id)["diff"].mean()

    def plot(
        self,
        bins: int = 10,
        cumulative: bool = False,
        ax: Axes | None = None,
        percentile_line: float | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        source_text: str | None = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the distribution of the average number of days between purchases per customer.

        Args:
            bins (int, optional): The number of bins to plot. Defaults to 10.
            cumulative (bool, optional): Whether to plot the cumulative distribution. Defaults to False.
            ax (Axes, optional): The Matplotlib axes to plot the graph on. Defaults to None.
            percentile_line (float, optional): The percentile to draw a line at. Defaults to None. When None then no
                line is drawn.
            title (str, optional): The title of the plot. Defaults to None.
            x_label (str, optional): The x-axis label. Defaults to None.
            y_label (str, optional): The y-axis label. Defaults to None.
            source_text (str, optional): The source text to add to the plot. Defaults to None.
            kwargs (dict[str, any]): Additional keyword arguments to pass to the plot

        Returns:
            SubplotBase: The Matplotlib axes of the plot
        """
        density = False
        if cumulative:
            density = True

        ax = self.purchase_dist_s.hist(
            bins=bins,
            cumulative=cumulative,
            ax=ax,
            density=density,
            color=COLORS["green"][500],
            **kwargs,
        )

        ax.xaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        ax = standard_graph_styles(ax)

        if cumulative:
            default_title = "Average Days Between Purchases cumulative Distribution"
            default_y_label = "Percentage of Customers"
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        else:
            default_title = "Average Days Between Purchases Distribution"
            default_y_label = "Number of Customers"
            ax.yaxis.set_major_formatter(lambda x, pos: human_format(x, pos, decimals=0))

        ax = gu.standard_graph_styles(
            ax,
            title=gu.not_none(title, default_title),
            y_label=gu.not_none(y_label, default_y_label),
            x_label=gu.not_none(x_label, "Average Number of Days Between Purchases"),
        )

        if percentile_line is not None:
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
            gu.add_source_text(ax=ax, source_text=source_text)

        gu.standard_tick_styles(ax)

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
    """

    def __init__(self, df: pd.DataFrame, churn_period: float) -> None:
        """Initialize the TransactionChurn class.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must have the columns customer_id
                and transaction_date.
            churn_period (float): The number of days to consider a customer churned.

        Raises:
            ValueError: If the dataframe does doesn't contain the columns customer_id and transaction_id.
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_date]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        purchase_dist_df = df[[cols.customer_id, cols.transaction_date]].copy()
        # Truncate the transaction_date to the day
        purchase_dist_df[cols.transaction_date] = df[cols.transaction_date].dt.floor("D")
        purchase_dist_df = purchase_dist_df.drop_duplicates()
        purchase_dist_df = purchase_dist_df.sort_values([cols.customer_id, cols.transaction_date])
        purchase_dist_df["transaction_number"] = purchase_dist_df.groupby(cols.customer_id).cumcount() + 1

        purchase_dist_df["last_transaction"] = (
            purchase_dist_df.groupby(cols.customer_id)[cols.transaction_date].shift(-1).isna()
        )
        purchase_dist_df["transaction_before_churn_window"] = purchase_dist_df[cols.transaction_date] < (
            purchase_dist_df[cols.transaction_date].max() - pd.Timedelta(days=churn_period)
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

        self.n_unique_customers = df[cols.customer_id].nunique()

    def plot(
        self,
        cumulative: bool = False,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        source_text: str | None = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plot the churn rate by number of purchases.

        Args:
            cumulative (bool, optional): Whether to plot the cumulative distribution. Defaults to False.
            ax (Axes, optional): The Matplotlib axes to plot the graph on. Defaults to None.
            title (str, optional): The title of the plot. Defaults to None.
            x_label (str, optional): The x-axis label. Defaults to None.
            y_label (str, optional): The y-axis label. Defaults to None.
            source_text (str, optional): The source text to add to the plot. Defaults to None.
            kwargs (dict[str, any]): Additional keyword arguments to pass to the plot function.

        Returns:
            SubplotBase: The Matplotlib axes of the plot
        """
        if cumulative:
            cumulative_churn_rate_s = self.purchase_dist_df["churned"].cumsum().div(self.n_unique_customers)
            ax = cumulative_churn_rate_s.plot.area(
                color=COLORS["green"][500],
                **kwargs,
            )
            ax.set_xlim(self.purchase_dist_df.index.min(), self.purchase_dist_df.index.max())
        else:
            ax = self.purchase_dist_df["churned_pct"].plot.bar(
                rot=0,
                color=COLORS["green"][500],
                width=0.8,
                **kwargs,
            )

        standard_graph_styles(
            ax,
            title=gu.not_none(title, "Churn Rate by Number of Purchases"),
            x_label=gu.not_none(x_label, "Number of Purchases"),
            y_label=gu.not_none(y_label, "% Churned"),
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        if source_text:
            gu.add_source_text(ax=ax, source_text=source_text)

        return ax
