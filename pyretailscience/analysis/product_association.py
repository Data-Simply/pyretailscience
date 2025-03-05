"""Product Association Rules Generation.

This module implements functionality for generating product association rules, a powerful technique in retail analytics
and market basket analysis.

Product association rules are used to uncover relationships between different products that customers tend to purchase
together. These rules provide valuable insights into consumer behavior and purchasing patterns, which can be leveraged
by retail businesses in various ways:

1. Cross-selling and upselling: By identifying products frequently bought together, retailers can make targeted product
   recommendations to increase sales and average order value.

2. Store layout optimization: Understanding product associations helps in strategic product placement within stores,
   potentially increasing impulse purchases and overall sales.

3. Inventory management: Knowing which products are often bought together aids in maintaining appropriate stock levels
   and predicting demand.

4. Marketing and promotions: Association rules can guide the creation ofeffective bundle offers and promotional
   campaigns.

5. Customer segmentation: Patterns in product associations can reveal distinct customer segments with specific
   preferences.

6. New product development: Insights from association rules can inform decisions about new product lines or features.

The module uses metrics such as support, confidence, and uplift to quantifythe strength and significance of product
associations:

- Support: The frequency of items appearing together in transactions.
- Confidence: The likelihood of buying one product given the purchase of another.
- Uplift: The increase in purchase probability of one product when another is bought.

By leveraging these association rules, retailers can make data-driven decisions to enhance customer experience, optimize
operations, and drive business growth.
"""

from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from tqdm import tqdm

from pyretailscience.options import get_option


class ProductAssociation:
    """A class for generating and analyzing product association rules.

    This class calculates association rules between products based on transaction data,
    helping to identify patterns in customer purchasing behavior.

    Args:
        df (pandas.DataFrame): The input DataFrame containing transaction data.
        value_col (str): The name of the column in the input DataFrame that contains
            the product identifiers.
        group_col (str, optional): The name of the column that identifies unique
            transactions or customers. Defaults to option column.column_id.
        target_item (str or None, optional): A specific product to focus the
            association analysis on. If None, associations for all products are
            calculated. Defaults to None.

    Attributes:
        df (pandas.DataFrame): A DataFrame containing the calculated association
            rules and their metrics.

    Example:
        >>> import pandas as pd
        >>> transaction_df = pd.DataFrame({
        ...     'customer_id': [1, 1, 2, 2, 3],
        ...     'product_id': ['A', 'B', 'B', 'C', 'A']
        ... })
        >>> pa = ProductAssociation(df=transaction_df, value_col='product_id', group_col='customer_id')
        >>> print(pa.df)  # View the calculated association rules

    Note:
        The resulting DataFrame (pa.df) contains the following columns:
        - product_1, product_2: The pair of products for which the association is calculated.
        - occurrences_1, occurrences_2: The number of transactions containing each product.
        - cooccurrences: The number of transactions containing both products.
        - support: The proportion of transactions containing both products.
        - confidence: The probability of buying product_2 given that product_1 was bought.
        - uplift: The ratio of the observed support to the expected support if the products were independent.

        The class uses efficient sparse matrix operations to handle large datasets and
        calculates associations for either pairs (2) or triples (3) of products, depending
        on the 'number_of_combinations' parameter in _calc_association.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str = get_option("column.customer_id"),
        target_item: str | None = None,
        number_of_combinations: Literal[2, 3] = 2,
        min_occurrences: int = 1,
        min_cooccurrences: int = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_uplift: float = 0.0,
        show_progress: bool = False,
    ) -> None:
        """Initialize the ProductAssociation object.

        Args:
            df (pandas.DataFrame): The input DataFrame containing transaction data.
            value_col (str): The name of the column in the input DataFrame that contains the product identifiers.
            group_col (str, optional): The name of the column that identifies unique transactions or customers. Defaults
                to option column.unit_spend.
            target_item (str or None, optional): A specific product to focus the association analysis on. If None,
                associations for all products are calculated. Defaults to None.
            number_of_combinations (int, optional): The number of products to consider in the association analysis. Can
                be either 2 or 3. Defaults to 2.
            min_occurrences (int, optional): The minimum number of occurrences required for each product in the
                association analysis. Defaults to 1. Must be at least 1.
            min_cooccurrences (int, optional): The minimum number of co-occurrences required for the product pairs in
                the association analysis. Defaults to 1. Must be at least 1.
            min_support (float, optional): The minimum support value required for the association rules. Defaults to
                0.0. Must be between 0 and 1.
            min_confidence (float, optional): The minimum confidence value required for the association rules. Defaults
                to 0.0. Must be between 0 and 1.
            min_uplift (float, optional): The minimum uplift value required for the association rules. Defaults to 0.0.
                Must be greater or equal to 0.
            show_progress (bool, optional): Whether to show a progress bar during the calculation. Defaults to False.

        Raises:
            ValueError: If the number of combinations is not 2 or 3, or if any of the minimum values are invalid.
            ValueError: If the minimum support, confidence, or uplift values are outside the valid range.
            ValueError: If the minimum occurrences or cooccurrences are less than 1.
            ValueError: If the input DataFrame does not contain the required columns or if they have null values.
        """
        required_cols = [group_col, value_col]
        missing_cols = set(required_cols) - set(df.columns)
        if len(missing_cols) > 0:
            msg = f"The following columns are required but missing: {missing_cols}"
            raise ValueError(msg)

        self.df = self._calc_association(
            df=df,
            value_col=value_col,
            group_col=group_col,
            target_item=target_item,
            number_of_combinations=number_of_combinations,
            min_occurrences=min_occurrences,
            min_cooccurrences=min_cooccurrences,
            min_support=min_support,
            min_confidence=min_confidence,
            min_uplift=min_uplift,
            show_progress=show_progress,
        )

    @staticmethod
    def _calc_association(  # noqa: C901 (ignore complexity) - Excluded due to min_* arguments checks
        df: pd.DataFrame,
        value_col: str,
        group_col: str = get_option("column.customer_id"),
        target_item: str | None = None,
        number_of_combinations: Literal[2, 3] = 2,
        min_occurrences: int = 1,
        min_cooccurrences: int = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_uplift: float = 0.0,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Calculate product association rules based on transaction data.

        This method calculates association rules between products based on transaction data,
        helping to identify patterns in customer purchasing behavior.

        Args:
            df (pandas.DataFrame): The input DataFrame containing transaction data.
            value_col (str): The name of the column in the input DataFrame that contains the product identifiers.
            group_col (str, optional): The name of the column that identifies unique transactions or customers. Defaults
                to option column.unit_spend.
            target_item (str or None, optional): A specific product to focus the association analysis on. If None,
                associations for all products are calculated. Defaults to None.
            number_of_combinations (int, optional): The number of products to consider in the association analysis. Can
                be either 2 or 3. Defaults to 2.
            min_occurrences (int, optional): The minimum number of occurrences required for each product in the
                association analysis. Defaults to 1. Must be at least 1.
            min_cooccurrences (int, optional): The minimum number of co-occurrences required for the product pairs in
                the association analysis. Defaults to 1. Must be at least 1.
            min_support (float, optional): The minimum support value required for the association rules. Defaults to
                0.0. Must be between 0 and 1.
            min_confidence (float, optional): The minimum confidence value required for the association rules. Defaults
                to 0.0. Must be between 0 and 1.
            min_uplift (float, optional): The minimum uplift value required for the association rules. Defaults to 0.0.
                Must be greater or equal to 0.
            show_progress (bool, optional): Whether to show a progress bar during the calculation. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the calculated association rules and their metrics.

        Raises:
            ValueError: If the number of combinations is not 2 or 3, or if any of the minimum values are invalid.
            ValueError: If the minimum support, confidence, or uplift values are outside the valid range.
            ValueError: If the minimum occurrences or cooccurrences are less than 1.

        Note:
            The resulting DataFrame contains the following columns:
            - product_1, product_2: The pair of products for which the association is calculated.
            - occurrences_1, occurrences_2: The number of transactions containing each product.
            - cooccurrences: The number of transactions containing both products.
            - support: The proportion of transactions containing both products.
            - confidence: The probability of buying product_2 given that product_1 was bought.
            - uplift: The ratio of the observed support to the expected support if the products were independent.

            The method uses efficient sparse matrix operations to handle large datasets and
            calculates associations for either pairs (2) or triples (3) of products, depending
            on the 'number_of_combinations' parameter.
        """
        if number_of_combinations not in [2, 3]:
            raise ValueError("Number of combinations must be either 2 or 3.")
        if min_occurrences < 1:
            raise ValueError("Minimum occurrences must be at least 1.")
        if min_cooccurrences < 1:
            raise ValueError("Minimum cooccurrences must be at least 1.")
        if min_support < 0.0 or min_support > 1.0:
            raise ValueError("Minimum support must be between 0 and 1.")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("Minimum confidence must be between 0 and 1.")
        if min_uplift < 0.0:
            raise ValueError("Minimum uplift must be greater or equal to 0.")

        unique_combo_df = df[[group_col, value_col]].drop_duplicates()
        unique_combo_df[value_col] = pd.Categorical(unique_combo_df[value_col], ordered=True)
        unique_combo_df[group_col] = pd.Categorical(unique_combo_df[group_col], ordered=True)

        sparse_matrix = csc_matrix(
            (
                [1] * len(unique_combo_df),
                (
                    unique_combo_df[group_col].cat.codes,
                    unique_combo_df[value_col].cat.codes,
                ),
            ),
        )

        row_count = sparse_matrix.shape[0]

        results = []

        occurrences = np.array(sparse_matrix.sum(axis=0)).flatten()
        occurence_prob = occurrences / row_count

        base_items = [target_item]
        if number_of_combinations == 2:  # noqa: PLR2004
            if target_item is None:
                base_items = unique_combo_df[value_col].cat.categories
            items = [([unique_combo_df[value_col].cat.categories.get_loc(cat)], cat) for cat in base_items]
        elif number_of_combinations == 3:  # noqa: PLR2004
            if target_item is None:
                base_items = sorted(combinations(unique_combo_df[value_col].cat.categories, 2))
            items = [
                ([unique_combo_df[value_col].cat.categories.get_loc(i) for i in cats], cats) for cats in base_items
            ]

        if show_progress:
            items = tqdm(items)

        cols_mask = np.zeros(sparse_matrix.shape[1], dtype=bool)

        for target_item_loc, item_2 in items:
            target_item_col_index = cols_mask.copy()
            target_item_col_index[target_item_loc] = True
            rows_with_target_item = sparse_matrix[:, target_item_col_index].getnnz(axis=1) == len(target_item_loc)

            cooccurrences = np.array(sparse_matrix[rows_with_target_item, :].sum(axis=0)).flatten()
            if (cooccurrences == 0).all():
                continue

            rows_with_target_item_sum = rows_with_target_item.sum()
            coocurrence_prob = cooccurrences / row_count

            target_prob = rows_with_target_item_sum / row_count
            expected_prob = target_prob * occurence_prob

            # TODO: Try to avoid constructing a pandas Dataframe
            pa_df = pd.DataFrame(
                {
                    f"{value_col}_1": [item_2] * sparse_matrix.shape[1],
                    f"{value_col}_2": unique_combo_df[value_col].cat.categories.values,
                    "occurrences_1": rows_with_target_item_sum,
                    "occurrences_2": occurrences,
                    "cooccurrences": cooccurrences,
                    "support": coocurrence_prob,
                    "confidence": cooccurrences / rows_with_target_item_sum,
                    "uplift": coocurrence_prob / expected_prob,
                },
            )

            excl_pairs_idx = (
                target_item_col_index
                | (pa_df["occurrences_1"] < min_occurrences)
                | (pa_df["occurrences_2"] < min_occurrences)
                | (pa_df["cooccurrences"] < min_cooccurrences)
                | (pa_df["support"] < min_support)
                | (pa_df["confidence"] < min_confidence)
                | (pa_df["uplift"] < min_uplift)
            )

            results.append(pa_df[~excl_pairs_idx])

        return pd.concat(results).sort_values([f"{value_col}_1", f"{value_col}_2"]).reset_index(drop=True)
