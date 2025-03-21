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

import ibis
import pandas as pd

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
    """

    _df: pd.DataFrame | None = None

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        value_col: str,
        group_col: str = get_option("column.customer_id"),
        target_item: str | None = None,
        min_occurrences: int = 1,
        min_cooccurrences: int = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_uplift: float = 0.0,
    ) -> None:
        """Initialize the ProductAssociation object.

        Args:
            df (pd.DataFrame | ibis.Table) : The input DataFrame or ibis Table containing transaction data.
            value_col (str): The name of the column in the input DataFrame that contains the product identifiers.
            group_col (str, optional): The name of the column that identifies unique transactions or customers. Defaults
                to option column.unit_spend.
            target_item (str or None, optional): A specific product to focus the association analysis on. If None,
                associations for all products are calculated. Defaults to None.
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

        self.table = self._calc_association(
            df=df,
            value_col=value_col,
            group_col=group_col,
            target_item=target_item,
            min_occurrences=min_occurrences,
            min_cooccurrences=min_cooccurrences,
            min_support=min_support,
            min_confidence=min_confidence,
            min_uplift=min_uplift,
        )

    @staticmethod
    def _calc_association(
        df: pd.DataFrame | ibis.Table,
        value_col: str,
        group_col: str = get_option("column.customer_id"),
        target_item: str | None = None,
        min_occurrences: int = 1,
        min_cooccurrences: int = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_uplift: float = 0.0,
    ) -> pd.DataFrame:
        """Calculate product association rules based on transaction data.

        This method calculates association rules between products based on transaction data,
        helping to identify patterns in customer purchasing behavior.

        Args:
            df (pd.DataFrame | ibis.Table) : The input DataFrame or ibis Table containing transaction data.
            value_col (str): The name of the column in the input DataFrame that contains the product identifiers.
            group_col (str, optional): The name of the column that identifies unique transactions or customers. Defaults
                to option column.unit_spend.
            target_item (str or None, optional): A specific product to focus the association analysis on. If None,
                associations for all products are calculated. Defaults to None.
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
        """
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

        if isinstance(df, pd.DataFrame):
            df = ibis.memtable(df)

        unique_transactions = df.select(df[group_col], df[value_col]).distinct()
        total_transactions = unique_transactions.alias("t")[group_col].nunique().name("total_count")

        product_occurrences = (
            unique_transactions.group_by(value_col)
            .aggregate(
                occurrences=lambda t: t[group_col].nunique(),
            )
            .mutate(occurrence_probability=lambda t: t.occurrences / total_transactions)
            .filter(lambda t: t.occurrences >= min_occurrences)
        )

        left_table = unique_transactions.rename({"item_1": value_col})
        right_table = unique_transactions.rename({"item_2": value_col})

        join_logic = [left_table[group_col] == right_table[group_col]]
        if target_item is None:
            join_logic.append(left_table["item_1"] < right_table["item_2"])
        else:
            join_logic.extend(
                [
                    left_table["item_1"] != right_table["item_2"],
                    left_table["item_1"] == target_item,
                ],
            )
        merged_df = left_table.join(
            right_table,
            predicates=join_logic,
            lname="",
            rname="{name}_right",
        )

        product_occurrences_1 = product_occurrences.rename(
            {"item_1": value_col, "occurrences_1": "occurrences", "occurrence_probability_1": "occurrence_probability"},
        )
        product_occurrences_2 = product_occurrences.rename(
            {"item_2": value_col, "occurrences_2": "occurrences", "occurrence_probability_2": "occurrence_probability"},
        )

        merged_df = merged_df.join(
            product_occurrences_1,
            predicates=[merged_df["item_1"] == product_occurrences_1["item_1"]],
        )

        merged_df = merged_df.join(
            product_occurrences_2,
            predicates=[merged_df["item_2"] == product_occurrences_2["item_2"]],
        )

        cooccurrences = merged_df.group_by(["item_1", "item_2"]).aggregate(cooccurrences=merged_df[group_col].nunique())
        cooccurrences = cooccurrences.mutate(
            support=cooccurrences.cooccurrences / total_transactions,
        )
        cooccurrences = cooccurrences.filter(
            (cooccurrences.cooccurrences >= min_cooccurrences) & (cooccurrences.support >= min_support),
        )

        product_occurrences_1_rename = product_occurrences.rename(
            {"item_1": value_col, "occurrences_1": "occurrences", "prob_1": "occurrence_probability"},
        )
        product_occurrences_2_rename = product_occurrences.rename(
            {"item_2": value_col, "occurrences_2": "occurrences", "prob_2": "occurrence_probability"},
        )

        product_pairs = cooccurrences.join(
            product_occurrences_1_rename,
            predicates=[cooccurrences["item_1"] == product_occurrences_1_rename["item_1"]],
        )
        product_pairs = product_pairs.join(
            product_occurrences_2_rename,
            predicates=[product_pairs["item_2"] == product_occurrences_2_rename["item_2"]],
        )

        product_pairs = product_pairs.mutate(
            confidence=product_pairs["cooccurrences"] / product_pairs["occurrences_1"],
            uplift=product_pairs["support"] / (product_pairs["prob_1"] * product_pairs["prob_2"]),
        )

        result = product_pairs.filter(product_pairs.uplift >= min_uplift)

        if target_item is None:
            col_order = [
                "item_1",
                "item_2",
                "occurrences_1",
                "occurrences_2",
                "cooccurrences",
                "support",
                "confidence",
                "uplift",
            ]
            inverse_pairs = result.mutate(
                item_1=result["item_2"],
                item_2=result["item_1"],
                occurrences_1=result["occurrences_2"],
                occurrences_2=result["occurrences_1"],
                prob_1=result["prob_2"],
                prob_2=result["prob_1"],
                confidence=result["cooccurrences"] / result["occurrences_2"],
            )
            result = result[col_order].union(inverse_pairs[col_order])

        result = result.filter(result.confidence >= min_confidence)
        final_result = result.order_by(["item_1", "item_2"])
        final_result = final_result.rename(
            {
                f"{value_col}_1": "item_1",
                f"{value_col}_2": "item_2",
            },
        )
        return final_result[
            [
                f"{value_col}_1",
                f"{value_col}_2",
                "occurrences_1",
                "occurrences_2",
                "cooccurrences",
                "support",
                "confidence",
                "uplift",
            ]
        ]

    @property
    def df(self) -> pd.DataFrame:
        """Returns the executed DataFrame."""
        if self._df is None:
            self._df = self.table.execute().reset_index(drop=True)
        return self._df
