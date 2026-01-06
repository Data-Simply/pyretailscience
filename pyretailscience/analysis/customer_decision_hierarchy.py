"""Customer Decision Hierarchy Analysis for Product Substitutability and Range Optimization.

## Business Context

Customer Decision Hierarchy (CDH) analysis reveals how customers perceive products
as substitutes or complements. This critical intelligence informs range planning,
assortment optimization, and delisting decisions by understanding which products
customers view as interchangeable versus essential variety.

## The Business Problem

Retailers often struggle with range rationalization decisions:
- Which products can be delisted without losing customers?
- When does variety add value versus create confusion?
- Which products are true substitutes in customers' minds?
- How to optimize shelf space without sacrificing choice?

CDH analysis answers these questions by analyzing actual switching behavior rather
than relying on product attributes or manager intuition.

## How It Works

The analysis examines customer purchase patterns to identify substitutability:
- Products rarely bought by the same customer → likely substitutes
- Products often bought by the same customer → complements or variety-seeking
- Uses Yule's Q coefficient to measure substitutability strength
- Creates hierarchical clusters showing substitution relationships

## Real-World Applications

1. **Range Rationalization**
   - Identify safe delisting candidates within substitute clusters
   - Maintain one option per cluster to preserve choice
   - Reduce SKU count while maintaining customer satisfaction

2. **New Product Introduction**
   - Understand which existing products new items might cannibalize
   - Position new products to fill gaps rather than duplicate
   - Predict source of volume for new launches

3. **Private Label Strategy**
   - Identify national brand products suitable for PL alternatives
   - Understand where PL can substitute vs. complement
   - Optimize PL/NB mix within categories

4. **Space Optimization**
   - Allocate more space to non-substitutable products
   - Reduce facings for products within same substitute cluster
   - Optimize variety/productivity trade-off

5. **Markdown Strategy**
   - Clear substitute products sequentially, not simultaneously
   - Understand which products can drive category traffic
   - Identify products that won't cannibalize when promoted

## Business Value

- **Efficient Assortment**: Reduce complexity without losing sales
- **Better Space Productivity**: Allocate space based on true variety value
- **Improved Margins**: Replace duplicative SKUs with unique offerings
- **Customer Satisfaction**: Maintain perceived choice while reducing confusion
- **Strategic Clarity**: Data-driven approach to range decisions
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from scipy.cluster.hierarchy import dendrogram, linkage

import pyretailscience.plots.styles.graph_utils as gu
from pyretailscience.analysis.demand.aids_estimator import AIDSEstimator
from pyretailscience.options import ColumnHelper, get_option
from pyretailscience.plots.styles.styling_helpers import PlotStyler


class CustomerDecisionHierarchy:
    """Analyzes product substitutability patterns to optimize retail assortments.

    The CustomerDecisionHierarchy class identifies which products customers view as
    substitutes versus essential variety. This enables data-driven range planning
    decisions that maintain customer choice while improving operational efficiency.

    ## Business Insight

    Traditional range planning often assumes products in the same category are
    substitutes (e.g., all yogurts are interchangeable). However, customer behavior
    reveals the truth: some customers always buy both Greek and regular yogurt
    (complements), while others switch between strawberry and raspberry flavors
    (substitutes).

    ## Substitutability Logic

    The analysis identifies substitutes through purchase patterns:
    - **High substitutability**: Customers buy product A OR product B, rarely both
    - **Low substitutability**: Customers often buy both A AND B
    - **Exclusion logic**: Products bought in same transaction can't be substitutes

    ## Decision Framework

    The hierarchy output guides range decisions:
    - **Tight clusters**: Strong substitutes - keep best performer
    - **Loose clusters**: Weak substitutes - maintain variety
    - **Separate branches**: Different needs - preserve both
    - **Isolated products**: Unique value - protect from delisting

    ## Example Use Case

    A supermarket analyzing yogurt finds:
    - Cluster 1: Store brand vanilla, strawberry, raspberry (substitutes)
    - Cluster 2: Greek plain, Greek honey (substitutes)
    - Separate branch: Kids' squeezable yogurt (unique need)

    Decision: Can reduce flavor variety in Cluster 1, maintain Greek options,
    must keep kids' yogurt despite low sales.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        product_col: str,
        exclude_same_transaction_products: bool = True,
        method: Literal["yules_q", "aids"] = "yules_q",
        random_state: int = 42,
        price_col: str | None = None,
        quantity_col: str | None = None,
        expenditure_col: str | None = None,
    ) -> None:
        """Initialize customer decision hierarchy analysis for range optimization.

        Args:
            df (pd.DataFrame): Transaction data with customer purchase history.
                Must contain: customer_id, transaction_id, and product identifier.
                For AIDS method, also requires price and quantity columns.
            product_col (str): Column containing products to analyze for substitutability
                (e.g., "product_name", "sku", "brand", "subcategory").
            exclude_same_transaction_products (bool, optional): Whether products bought
                together in one transaction should be considered non-substitutes.
                True = If customer buys milk and eggs together, they're not substitutes.
                False = Include all purchase patterns.
                Defaults to True (recommended for most retail contexts).
                Only applies to "yules_q" method.
            method (Literal["yules_q", "aids"], optional): Statistical method for measuring
                substitutability:
                - "yules_q": Correlation-based measure using purchase patterns (default)
                - "aids": Causal elasticity-based measure using Almost Ideal Demand System
                Defaults to "yules_q".
            random_state (int, optional): Seed for reproducible clustering results.
                Important for consistent range planning decisions. Defaults to 42.
            price_col (str | None, optional): Column containing prices. Required for AIDS method.
                Defaults to None.
            quantity_col (str | None, optional): Column containing quantities. Required for AIDS method.
                Defaults to None.
            expenditure_col (str | None, optional): Column containing total expenditure.
                Optional for AIDS method. Defaults to None.

        Raises:
            ValueError: If required columns are missing from the dataframe.
            ValueError: If AIDS method is selected without required price/quantity columns.

        Business Example:
            >>> # Analyze substitutability using correlation (Yule's Q)
            >>> cdh = CustomerDecisionHierarchy(
            ...     df=transactions,
            ...     product_col="brand_flavor",
            ...     exclude_same_transaction_products=True,
            ...     method="yules_q"
            ... )
            >>> # Analyze substitutability using causal elasticities (AIDS)
            >>> cdh = CustomerDecisionHierarchy(
            ...     df=transactions,
            ...     product_col="brand",
            ...     method="aids",
            ...     price_col="unit_price",
            ...     quantity_col="units_sold"
            ... )
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_id, product_col]

        # Validate method-specific requirements
        if method == "aids":
            if price_col is None or quantity_col is None:
                msg = "AIDS method requires price_col and quantity_col to be specified"
                raise ValueError(msg)
            required_cols.extend([price_col, quantity_col])
            if expenditure_col is not None:
                required_cols.append(expenditure_col)

        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.random_state = random_state
        self.product_col = product_col
        self.method = method
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.expenditure_col = expenditure_col

        if method == "yules_q":
            self.pairs_df = self._get_pairs(df, exclude_same_transaction_products, product_col)
            self.distances = self._calculate_distances(method=method)
        elif method == "aids":
            self.aids_estimator = self._fit_aids_model(df)
            self.distances = self._calculate_distances(method=method)
        else:
            msg = f"Invalid method: {method}. Must be 'yules_q' or 'aids'"
            raise ValueError(msg)

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

    def _fit_aids_model(self, df: pd.DataFrame) -> AIDSEstimator:
        """Fit AIDS model to compute elasticity-based distances.

        Args:
            df (pd.DataFrame): Transaction data with prices and quantities.

        Returns:
            AIDSEstimator: Fitted AIDS estimator.
        """
        # Aggregate data to product level for AIDS estimation
        # AIDS requires panel data structure: observations x products
        estimator = AIDSEstimator(
            df=df,
            product_col=self.product_col,
            price_col=self.price_col,
            quantity_col=self.quantity_col,
            expenditure_col=self.expenditure_col,
            price_index_method="tornqvist",
            enforce_constraints=True,
        )
        estimator.fit()
        return estimator

    def _get_aids_distances(self) -> np.ndarray:
        """Calculate distances based on AIDS elasticities.

        Uses cross-price elasticities to define product distances:
        - Large negative elasticity = complements (far apart)
        - Near zero elasticity = independent (moderate distance)
        - Large positive elasticity = substitutes (close together)

        Returns:
            np.ndarray: Distance matrix based on elasticity patterns.
        """
        elasticities = self.aids_estimator.get_elasticities()

        # Extract cross-price elasticity matrix (exclude expenditure column)
        price_cols = [col for col in elasticities.columns if col.endswith("_price")]
        elasticity_matrix = elasticities[price_cols].values

        # Convert elasticities to distances (only for off-diagonal elements)
        n_products = len(elasticity_matrix)
        distances = np.zeros((n_products, n_products))

        for i in range(n_products):
            for j in range(n_products):
                if i != j:
                    elasticity = elasticity_matrix[i, j]
                    if elasticity > 0:
                        # Substitutes: closer distance for higher elasticity
                        distances[i, j] = 1.0 / (1.0 + elasticity)
                    else:
                        # Complements: farther distance for more negative elasticity
                        distances[i, j] = 1.0 + abs(elasticity)

        # Normalize distances to [0, 1] range for consistency with Yule's Q
        # Only normalize off-diagonal elements
        non_diag_distances = distances[~np.eye(n_products, dtype=bool)]
        if len(non_diag_distances) > 0:
            min_dist = non_diag_distances.min()
            max_dist = non_diag_distances.max()
            if max_dist > min_dist:
                # Normalize only off-diagonal elements
                for i in range(n_products):
                    for j in range(n_products):
                        if i != j:
                            distances[i, j] = (distances[i, j] - min_dist) / (max_dist - min_dist)

        return distances

    def _calculate_distances(
        self,
        method: Literal["yules_q", "aids"],
    ) -> np.ndarray:
        """Calculates distances between items using the specified method.

        Args:
            method (Literal["yules_q", "aids"]): The method to use for calculating distances.

        Raises:
            ValueError: If the method is not valid.

        Returns:
            np.ndarray: Distance matrix between products.
        """
        # Check method is valid
        if method == "yules_q":
            distances = self._get_yules_q_distances()
        elif method == "aids":
            distances = self._get_aids_distances()
        else:
            raise ValueError("Method must be 'yules_q' or 'aids'")

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

        if self.method == "yules_q":
            labels = self.pairs_df[self.product_col].cat.categories
        elif self.method == "aids":
            labels = self.aids_estimator.products
        else:
            # This path should ideally not be reached due to validation in __init__
            labels = []

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
