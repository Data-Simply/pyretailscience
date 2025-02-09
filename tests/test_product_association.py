"""Tests for the ProductAssociation module."""

import pandas as pd
import pytest

from pyretailscience.options import ColumnHelper
from pyretailscience.product_association import ProductAssociation

cols = ColumnHelper()


class TestProductAssociations:
    """Tests for the ProductAssociations class."""

    @pytest.fixture()
    def transactions_df(self) -> pd.DataFrame:
        """Return a sample DataFrame for testing."""
        # fmt: off
        return pd.DataFrame({
            cols.transaction_id: [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5],
            "product": ["milk", "bread", "fruit", "butter", "eggs", "fruit", "beer", "diapers",
                        "milk", "bread", "butter", "eggs", "fruit", "bread"],
        })
        # fmt: on

    @pytest.fixture()
    def expected_results_single_items_df(self) -> pd.DataFrame:
        """Return the expected results for the single items association analysis."""
        # fmt: off
        return pd.DataFrame(
                {
                    "product_1": [
                        "beer", "bread", "bread", "bread", "bread", "butter", "butter", "butter", "butter", "diapers",
                        "eggs", "eggs", "eggs", "eggs", "fruit", "fruit", "fruit", "fruit", "milk", "milk", "milk",
                        "milk",
                    ],
                    "product_2": [
                        "diapers", "butter", "eggs", "fruit", "milk", "bread", "eggs", "fruit", "milk", "beer", "bread",
                        "butter", "fruit", "milk", "bread", "butter", "eggs", "milk", "bread", "butter", "eggs",
                        "fruit",
                    ],
                    "occurrences_1": [1, 3, 3, 3, 3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2],
                    "occurrences_2": [1, 2, 2, 3, 2, 3, 2, 3, 2, 1, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 3],
                    "cooccurrences": [1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2],
                    "support": [
                        0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4,
                        0.2, 0.2, 0.4,
                    ],
                    "confidence": [
                        1.0, 0.333333, 0.333333, 0.666667, 0.666667, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5,
                        0.666667, 0.666667, 0.666667, 0.666667, 1.0, 0.5, 0.5, 1.0,
                    ],
                    "uplift": [
                        5.0, 0.833333, 0.833333, 1.111111, 1.666667, 0.833333, 2.5, 1.666667, 1.25, 5.0, 0.833333, 2.5,
                        1.666667, 1.25, 1.111111, 1.666667, 1.666667, 1.666667, 1.666667, 1.25, 1.25, 1.666667,
                    ],
                },
        )
        # fmt: on

    @pytest.fixture()
    def expected_results_pair_items_df(self) -> pd.DataFrame:
        """Return the expected results for the pair items association analysis."""
        # fmt: off
        return pd.DataFrame(
            {
                "product_1": [
                    ("bread", "butter"), ("bread", "butter"), ("bread", "butter"), ("bread", "eggs"), ("bread", "eggs"),
                    ("bread", "eggs"), ("bread", "fruit"), ("bread", "fruit"), ("bread", "fruit"), ("bread", "milk"),
                    ("bread", "milk"), ("bread", "milk"), ("butter", "eggs"), ("butter", "eggs"), ("butter", "eggs"),
                    ("butter", "fruit"), ("butter", "fruit"), ("butter", "fruit"), ("butter", "milk"),
                    ("butter", "milk"), ("butter", "milk"), ("eggs", "fruit"), ("eggs", "fruit"), ("eggs", "fruit"),
                    ("eggs", "milk"), ("eggs", "milk"), ("eggs", "milk"), ("fruit", "milk"), ("fruit", "milk"),
                    ("fruit", "milk"),
                ],
                "product_2": [
                    "eggs", "fruit", "milk", "butter", "fruit", "milk", "butter", "eggs", "milk", "butter", "eggs",
                    "fruit", "bread", "fruit", "milk", "bread", "eggs", "milk", "bread", "eggs", "fruit", "bread",
                    "butter", "milk", "bread", "butter", "fruit", "bread", "butter", "eggs",
                ],
                "occurrences_1": [
                    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                ],
                "occurrences_2": [
                    2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 3, 2, 2,
                ],
                "cooccurrences": [
                    1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1,
                ],
                "support": [
                    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2,
                    0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2,
                ],
                "confidence": [
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 1.0,
                    1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5,
                ],
                "uplift": [
                    2.5, 1.666667, 2.5, 2.5, 1.666667, 2.5, 1.25, 1.25, 2.5, 1.25, 1.25, 1.666667, 0.833333, 1.666667,
                    1.25, 0.833333, 2.5, 1.25, 1.666667, 2.5, 1.666667, 0.833333, 2.5, 1.25, 1.666667, 2.5, 1.666667,
                    1.666667, 1.25,1.25,
                ],
            },
        )
        # fmt: on

    def test_calc_association_all_single_items(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules for a single item versus another of item for all items."""
        associations_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
        )

        pd.testing.assert_frame_equal(associations_df, expected_results_single_items_df)

    def test_calc_association_target_single_items(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules for target single item versus another of item."""
        target_item = "bread"

        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            target_item=target_item,
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_single_items_df[expected_results_single_items_df["product_1"] == target_item].reset_index(
                drop=True,
            ),
        )

    def test_calc_association_all_pair_items(self, transactions_df, expected_results_pair_items_df):
        """Test calculating association rules for a pairs of items versus another item for all items."""
        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            number_of_combinations=3,
        )

        pd.testing.assert_frame_equal(calc_df, expected_results_pair_items_df)

    def test_calc_association_target_pair_items(self, transactions_df, expected_results_pair_items_df):
        """Test calculating association rules for a target pairs of items versus another item."""
        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            number_of_combinations=3,
            target_item=("bread", "butter"),
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_pair_items_df[
                expected_results_pair_items_df["product_1"] == ("bread", "butter")
            ].reset_index(drop=True),
        )

    def test_calc_association_min_occurrences(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_occurrences = 2

        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_occurrences=min_occurrences,
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_single_items_df[
                (expected_results_single_items_df["occurrences_1"] >= min_occurrences)
                & (expected_results_single_items_df["occurrences_2"] >= min_occurrences)
            ].reset_index(drop=True),
        )

    def test_calc_association_min_cooccurrences(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_cooccurrences = 2

        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_cooccurrences=min_cooccurrences,
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_single_items_df[
                (expected_results_single_items_df["cooccurrences"] >= min_cooccurrences)
            ].reset_index(drop=True),
        )

    def test_calc_association_min_support(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_support = 0.25

        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_support=min_support,
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_single_items_df[(expected_results_single_items_df["support"] >= min_support)].reset_index(
                drop=True,
            ),
        )

    def test_calc_association_min_confidence(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_confidence = 0.25

        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_confidence=min_confidence,
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_single_items_df[
                (expected_results_single_items_df["confidence"] >= min_confidence)
            ].reset_index(drop=True),
        )

    def test_calc_association_min_uplift(self, transactions_df, expected_results_single_items_df):
        """Test calculating association rules with a min occurrences level."""
        min_uplift = 1

        calc_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col=cols.transaction_id,
            min_uplift=min_uplift,
        )

        pd.testing.assert_frame_equal(
            calc_df,
            expected_results_single_items_df[(expected_results_single_items_df["uplift"] >= min_uplift)].reset_index(
                drop=True,
            ),
        )

    def test_calc_association_invalid_number_of_combinations(self, transactions_df):
        """Test calculating association rules with an invalid number of combinations."""
        with pytest.raises(ValueError, match="Number of combinations must be either 2 or 3."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                number_of_combinations=4,
            )
        with pytest.raises(ValueError, match="Number of combinations must be either 2 or 3."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                number_of_combinations=1,
            )

    def test_calc_association_invalid_min_occurrences(self, transactions_df):
        """Test calculating association rules with an invalid minimum occurrences value."""
        with pytest.raises(ValueError, match="Minimum occurrences must be at least 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_occurrences=0,
            )

    def test_calc_association_invalid_min_cooccurrences(self, transactions_df):
        """Test calculating association rules with an invalid minimum cooccurrences value."""
        with pytest.raises(ValueError, match="Minimum cooccurrences must be at least 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_cooccurrences=0,
            )

    def test_calc_association_min_support_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum support range."""
        with pytest.raises(ValueError, match="Minimum support must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_support=-0.1,
            )
        with pytest.raises(ValueError, match="Minimum support must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_support=1.1,
            )

    def test_calc_association_min_confidence_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum confidence range."""
        with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_confidence=-0.1,
            )
        with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_confidence=1.1,
            )

    def test_calc_association_min_uplift_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum uplift range."""
        with pytest.raises(ValueError, match="Minimum uplift must be greater or equal to 0."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col=cols.transaction_id,
                min_uplift=-0.1,
            )
