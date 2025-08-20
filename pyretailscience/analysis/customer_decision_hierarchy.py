"""This module contains the RangePlanning class for performing customer decision hierarchy analysis."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from scipy.cluster.hierarchy import dendrogram, linkage

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.plots.styles.styling_helpers import PlotStyler


class CustomerDecisionHierarchy:
    """A class to perform customer decision hierarchy analysis using the Customer Decision Hierarchy method."""

    def __init__(
        self,
        df: pd.DataFrame,
        product_col: str,
        exclude_same_transaction_products: bool = True,
        method: Literal["yules_q"] = "yules_q",
        random_state: int = 42,
    ) -> None:
        """Initializes the RangePlanning object.

        Args:
            df (pd.DataFrame): The input dataframe containing transaction data. The dataframe must have the columns
                customer_id, transaction_id, product_name.
            product_col (str): The name of the column containing the product or category names.
            exclude_same_transaction_products (bool, optional): Flag indicating whether to exclude products found in
                the same transaction from a customer's distinct list of products bought. The idea is that if a
                customer bought two products in the same transaction they can't be substitutes for that customer.
                Thus they should be excluded from the analysis. Defaults to True.
            method (Literal["yules_q"], optional): The method to use for calculating distances.
                Defaults to "yules_q".
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If the dataframe does not have the require columns.

        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_id, product_col]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.random_state = random_state
        self.product_col = product_col
        self.pairs_df = self._get_pairs(df, exclude_same_transaction_products, product_col)
        self.distances = self._calculate_distances(method=method)

    @staticmethod
    def _get_pairs(df: pd.DataFrame, exclude_same_transaction_products: bool, product_col: str) -> pd.DataFrame:
        cols = ColumnHelper()
        if exclude_same_transaction_products:
            pairs_df = df[[cols.customer_id, cols.transaction_id, product_col]].drop_duplicates()
            pairs_to_exclude_df = (
                pairs_df.groupby(cols.transaction_id)
                .filter(lambda x: len(x) > 1)[[cols.customer_id, product_col]]
                .drop_duplicates()
            )
            # Drop all rows from pairs_df where customer_id and product_name are in pairs_to_exclude_df
            pairs_df = pairs_df.merge(
                pairs_to_exclude_df,
                on=[cols.customer_id, product_col],
                how="left",
                indicator=True,
            )
            pairs_df = pairs_df[pairs_df["_merge"] == "left_only"][[cols.customer_id, product_col]].drop_duplicates()
        else:
            pairs_df = df[[cols.customer_id, product_col]].drop_duplicates()

        return pairs_df.reset_index(drop=True).astype("category")

    @staticmethod
    def _calculate_yules_q(bought_product_1: np.array, bought_product_2: np.array) -> float:
        """Calculates the Yule's Q coefficient between two binary arrays.

        Args:
            bought_product_1 (np.array): Binary array representing the first bought product. Each element is 1 if the
                customer bought the product and 0 if they didn't.
            bought_product_2 (np.array): Binary array representing the second bought product. Each element is 1 if the
                customer bought the product and 0 if they didn't.

        Returns:
            float: The Yule's Q coefficient.

        Raises:
            ValueError: If the lengths of `bought_product_1` and `bought_product_2` are not the same.
            ValueError: If `bought_product_1` or `bought_product_2` is not a boolean array.

        """
        if len(bought_product_1) != len(bought_product_2):
            raise ValueError("The bought_product_1 and bought_product_2 must be the same length")
        if len(bought_product_1) == 0:
            return 0.0
        if bought_product_1.dtype != bool or bought_product_2.dtype != bool:
            raise ValueError("The bought_product_1 and bought_product_2 must be boolean arrays")

        a = np.count_nonzero(bought_product_1 & bought_product_2)
        b = np.count_nonzero(bought_product_1 & ~bought_product_2)
        c = np.count_nonzero(~bought_product_1 & bought_product_2)
        d = np.count_nonzero(~bought_product_1 & ~bought_product_2)

        # Calculate Yule's Q coefficient
        q = (a * d - b * c) / (a * d + b * c)

        return q  # noqa: RET504

    def _get_yules_q_distances(self) -> float:
        """Calculate the Yules Q distances between pairs of products.

        Returns:
            float: The Yules Q distances between pairs of products.
        """
        from scipy.sparse import csr_matrix

        # Create a sparse matrix where the rows are the customers and the columns are the products
        # The values are True if the customer bought the product and False if they didn't
        product_matrix = csr_matrix(
            (
                [True] * len(self.pairs_df),
                (
                    self.pairs_df[self.product_col].cat.codes,
                    self.pairs_df[get_option("column.customer_id")].cat.codes,
                ),
            ),
            dtype=bool,
        )

        # Calculate the number of customers and products
        n_products = product_matrix.shape[0]

        # Create an empty matrix to store the yules q values
        yules_q_matrix = np.zeros((n_products, n_products), dtype=float)

        # Loop through each pair of products
        for i in range(n_products):
            arr_i = product_matrix[i].toarray()
            for j in range(i + 1, n_products):
                # Calculate the yules q value for the pair of products
                arr_j = product_matrix[j].toarray()
                yules_q_dist = 1 - self._calculate_yules_q(arr_i, arr_j)

                # Store the yules q value in the matrix
                yules_q_matrix[i, j] = yules_q_dist
                yules_q_matrix[j, i] = yules_q_dist

        # Normalize the yules q values to be between 0 and 1 and return
        return (yules_q_matrix + 1) / 2

    def _calculate_distances(
        self,
        method: Literal["yules_q"],
    ) -> None:
        """Calculates distances between items using the specified method.

        Args:
            method (Literal["yules_q"], optional): The method to use for calculating distances.

        Raises:
            ValueError: If the method is not valid.

        Returns:
            None
        """
        # Check method is valid
        if method == "yules_q":
            distances = self._get_yules_q_distances()
        else:
            raise ValueError("Method must be 'yules_q'")

        return distances

    def plot(
        self,
        title: str = "Customer Decision Hierarchy",
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        figsize: tuple[int, int] | None = None,
        source_text: str | None = None,
        **kwargs: dict[str, any],
    ) -> SubplotBase:
        """Plots the customer decision hierarchy dendrogram.

        Args:
            title (str, optional): The title of the plot. Defaults to None.
            x_label (str, optional): The label for the x-axis. Defaults to None.
            y_label (str, optional): The label for the y-axis. Defaults to None.
            ax (Axes, optional): The matplotlib Axes object to plot on. Defaults to None.
            figsize (tuple[int, int], optional): The figure size. Defaults to None.
            source_text (str, optional): The source text to annotate on the plot. Defaults to None.
            **kwargs (dict[str, any]): Additional keyword arguments to pass to the dendrogram function.

        Returns:
            SubplotBase: The matplotlib SubplotBase object.
        """
        linkage_matrix = linkage(self.distances, method="ward")
        labels = self.pairs_df[self.product_col].cat.categories

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        orientation = kwargs.get("orientation", "top")
        default_x_label, default_y_label = (
            ("Products", "Distance") if orientation in ["top", "bottom"] else ("Distance", "Products")
        )

        gu.standard_graph_styles(
            ax=ax,
            title=title,
            x_label=gu.not_none(x_label, default_x_label),
            y_label=gu.not_none(y_label, default_y_label),
        )

        # Set the y label to be on the right side of the plot
        if orientation == "left":
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        elif orientation == "bottom":
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        dendrogram(linkage_matrix, labels=labels, ax=ax, **kwargs)
        styler = PlotStyler()
        styler.apply_ticks(ax)

        # Rotate the x-axis labels if they are too long
        if orientation in ["top", "bottom"]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Set the font properties for the tick labels
        gu.standard_tick_styles(ax)

        if source_text is not None:
            gu.add_source_text(ax=ax, source_text=source_text)

        return ax
