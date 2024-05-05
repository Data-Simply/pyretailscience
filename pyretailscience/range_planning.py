from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from scipy.cluster.hierarchy import dendrogram, linkage

import pyretailscience.style.graph_utils as gu
from pyretailscience.data.contracts import CustomContract, build_expected_columns, build_non_null_columns
from pyretailscience.style.graph_utils import GraphStyles as gs


class CustomerDecisionHierarchy:
    def __init__(
        self,
        df: pd.DataFrame,
        exclude_same_transaction_products: bool = True,
        method: Literal["truncated_svd", "yules_q"] = "truncated_svd",
        min_var_explained: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initializes the RangePlanning object.

        Args:
            df (pd.DataFrame): The input dataframe containing transaction data. The dataframe must comply with the
                TransactionItemLevelContract or the TransactionLevelContract.
            exclude_same_transaction_products (bool, optional): Flag indicating whether to exclude products found in
                the same transaction from a customer's distinct list of products bought. The idea is that if a
                customer bought two products in the same transaction they can't be substitutes for that customer.
                Thus they should be excluded from the analysis. Defaults to True.
            method (Literal["truncated_svd", "yules_q"], optional): The method to use for calculating distances.
                Defaults to "truncated_svd".
            min_var_explained (float, optional): The minimum variance explained required for truncated SVD method.
                Only applicable if method is "truncated_svd". Defaults to 0.8.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If the dataframe does not comply with the TransactionItemLevelContract.

        """
        cdh_contract = CustomContract(
            df,
            basic_expectations=build_expected_columns(columns=["customer_id", "transaction_id", "product_name"]),
            extended_expectations=build_non_null_columns(columns=["customer_id", "transaction_id", "product_name"]),
        )

        if cdh_contract.validate() is False:
            raise ValueError(
                "The dataframe requires the columns 'customer_id', 'transaction_id', and 'product_name' and they must "
                "be non-null"
            )

        self.random_state = random_state
        self.pairs_df = self._get_pairs(df, exclude_same_transaction_products)
        self.distances = self._calculate_distances(method=method, min_var_explained=min_var_explained)

    @staticmethod
    def _get_pairs(df: pd.DataFrame, exclude_same_transaction_products: bool) -> pd.DataFrame:
        if exclude_same_transaction_products:
            pairs_df = df[["customer_id", "transaction_id", "product_name"]].drop_duplicates()
            pairs_to_exclude_df = (
                pairs_df.groupby("transaction_id")
                .filter(lambda x: len(x) > 1)[["customer_id", "product_name"]]
                .drop_duplicates()
            )
            # Drop all rows from pairs_df where customer_id and product_name are in pairs_to_exclude_df
            pairs_df = pairs_df.merge(
                pairs_to_exclude_df, on=["customer_id", "product_name"], how="left", indicator=True
            )
            pairs_df = pairs_df[pairs_df["_merge"] == "left_only"][["customer_id", "product_name"]].drop_duplicates()
        else:
            pairs_df = df[["customer_id", "product_name"]].drop_duplicates()

        pairs_df = pairs_df.reset_index(drop=True).astype("category")

        return pairs_df

    def _get_truncated_svd_distances(self, min_var_explained: float = 0.8) -> np.array:
        """
        Calculate the truncated SVD distances for the given pairs dataframe.

        Args:
            min_var_explained (float): The minimum variance explained required.

        Returns:
            np.array: The normalized matrix of truncated SVD distances.
        """
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD

        sparse_matrix = csr_matrix(
            (
                [1] * len(self.pairs_df),
                (
                    self.pairs_df["product_name"].cat.codes,
                    self.pairs_df["customer_id"].cat.codes,
                ),
            )
        )

        n_products = sparse_matrix.shape[0]
        svd = TruncatedSVD(n_components=n_products, random_state=self.random_state)
        svd.fit(sparse_matrix)
        cuml_var = np.cumsum(svd.explained_variance_ratio_)

        req_n_components = np.argmax(cuml_var >= min_var_explained) + 1

        reduced_matrix = TruncatedSVD(n_components=req_n_components, random_state=self.random_state).fit_transform(
            sparse_matrix
        )
        norm_matrix = reduced_matrix / np.linalg.norm(reduced_matrix, axis=1, keepdims=True)

        return norm_matrix

    @staticmethod
    def _calculate_yules_q(bought_product_1: np.array, bought_product_2: np.array):
        """
        Calculates the Yule's Q coefficient between two binary arrays.

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

        q = (a * d - b * c) / (a * d + b * c)

        return q

    def _get_yules_q_distances(self) -> float:
        """
        Calculate the Yules Q distances between pairs of products.

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
                    self.pairs_df["product_name"].cat.codes,
                    self.pairs_df["customer_id"].cat.codes,
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

        # Normalize the yules q values to be between 0 and 1
        yules_q_matrix = (yules_q_matrix + 1) / 2

        return yules_q_matrix

    def _calculate_distances(
        self,
        method: Literal["truncated_svd", "yules_q"],
        min_var_explained: float,
    ) -> None:
        """Calculates distances between items using the specified method.

        Args:
            method (Literal["truncated_svd", "yules_q"], optional): The method to use for calculating distances.
            min_var_explained (float, optional): The minimum variance explained required for truncated SVD method.
                Only applicable if method is "truncated_svd".

        Raises:
            ValueError: If the method is not valid.

        Returns:
            None
        """
        # Check method is valid
        if method == "truncated_svd":
            distances = self._get_truncated_svd_distances(min_var_explained=min_var_explained)
        elif method == "yules_q":
            distances = self._get_yules_q_distances()
        else:
            raise ValueError("Method must be 'truncated_svd' or 'yules_q'")

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
        """
        Plots the range planning dendrogram.

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
        labels = self.pairs_df["product_name"].cat.categories

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        orientation = kwargs.get("orientation", "top")
        if orientation in ["top", "bottom"]:
            default_x_label = "Products"
            default_y_label = "Distance"
        else:
            default_x_label = "Distance"
            default_y_label = "Products"

        ax.set_title(
            title,
            fontsize=gs.DEFAULT_TITLE_FONT_SIZE,
            pad=15,
        )
        ax.set_xlabel(
            gu.not_none(y_label, default_x_label),
            fontsize=gs.DEFAULT_AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )
        ax.set_ylabel(
            gu.not_none(x_label, default_y_label),
            fontsize=gs.DEFAULT_AXIS_LABEL_FONT_SIZE,
            labelpad=10,
        )

        # Set the y label to be on the right side of the plot
        if orientation == "left":
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if orientation == "bottom":
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        dendrogram(linkage_matrix, labels=labels, ax=ax, **kwargs)

        if source_text is not None:
            plt.draw()

            y_axis_offset = -0.1
            if orientation != "bottom":
                bbox_height = ax.get_xaxis().get_tightbbox().height
                axis_height = bbox_height / ax.figure.dpi / ax.figure.get_figheight()
                y_axis_offset = min(-axis_height * 1.5, -0.1)

            x_axis_offset = -0.1
            if orientation == "right":
                bbox_width = ax.get_yaxis().get_tightbbox().width
                axis_width = bbox_width / ax.figure.dpi / ax.figure.get_figwidth()
                x_axis_offset = min(-axis_width * 1.25, -0.1)

            ax.annotate(
                source_text,
                xy=(x_axis_offset, y_axis_offset),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=gs.DEFAULT_SOURCE_FONT_SIZE,
            )

        if orientation in ["top", "bottom"]:
            ax.xaxis.set_tick_params(labelsize=gs.DEFAULT_TICK_LABEL_FONT_SIZE)
        else:
            ax.yaxis.set_tick_params(labelsize=gs.DEFAULT_TICK_LABEL_FONT_SIZE)

        return ax
