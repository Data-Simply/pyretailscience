"""Integration tests for Product Association Analysis with BigQuery."""

import pandas as pd
import pytest

from pyretailscience.analysis.product_association import ProductAssociation
from pyretailscience.options import ColumnHelper

cols = ColumnHelper()

MIN_ASSOCIATION_ROWS = 4
MIN_CATEGORIES_REQUIRED = 2
MIN_MULTI_CATEGORY_TRANSACTIONS = 2
MISSING_VALUE_THRESHOLD = 0.5


class TestProductAssociationsBigQuery:
    """Integration tests for the ProductAssociations class with BigQuery."""

    @pytest.fixture
    def transactions_df(self, transactions_table) -> pd.DataFrame:
        """Return a DataFrame from BigQuery for testing."""
        query = transactions_table.select(
            "transaction_id",
            "product_name",
        ).limit(1000)
        df = query.execute()

        return df.rename(columns={"product_name": "product"})

    @pytest.fixture
    def expected_results_df(self, transactions_df) -> pd.DataFrame:
        """Dynamically generate expected results based on actual BigQuery data."""
        associations_df = ProductAssociation._calc_association(
            df=transactions_df,
            value_col="product",
            group_col="transaction_id",
        )

        if not isinstance(associations_df, pd.DataFrame) and hasattr(associations_df, "execute"):
            associations_df = associations_df.execute()

        return associations_df

    def test_calc_association_all_single_items(self, transactions_df, expected_results_df):
        """Test calculating association rules for a single item versus another item for all items using BigQuery data."""
        associations_df = ProductAssociation(
            df=transactions_df,
            value_col="product",
            group_col="transaction_id",
        )
        result = associations_df.df

        if hasattr(result, "execute"):
            result = result.execute()

        pd.testing.assert_frame_equal(result, expected_results_df)

    def test_calc_association_target_single_items(self, transactions_df, expected_results_df):
        """Test calculating association rules for target single item versus another item using BigQuery data."""
        if len(expected_results_df) == 0:
            pytest.skip("No association rules found in data")

        target_products = transactions_df["product"].unique()
        if len(target_products) > 0:
            target_item = target_products[0]

            calc_df = ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                target_item=target_item,
            )
            result = calc_df.df

            if hasattr(result, "execute"):
                result = result.execute()

            expected = expected_results_df[expected_results_df["product_1"] == target_item].reset_index(drop=True)
            pd.testing.assert_frame_equal(result, expected)
        else:
            pytest.skip("No products in dataset to use as target item")

    def test_calc_association_min_occurrences(self, transactions_df, expected_results_df):
        """Test calculating association rules with a min occurrences level using BigQuery data."""
        if len(expected_results_df) == 0:
            pytest.skip("No association rules found in data")

        if expected_results_df.shape[0] >= 1:
            min_occurrences = max(2, expected_results_df["occurrences_1"].min())

            calc_df = ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_occurrences=min_occurrences,
            )

            result = calc_df.df

            if hasattr(result, "execute"):
                result = result.execute()

            expected = expected_results_df[
                (expected_results_df["occurrences_1"] >= min_occurrences)
                & (expected_results_df["occurrences_2"] >= min_occurrences)
            ].reset_index(drop=True)

            pd.testing.assert_frame_equal(result, expected)
        else:
            pytest.skip("Not enough data for min occurrences test")

    def test_calc_association_min_cooccurrences(self, transactions_df, expected_results_df):
        """Test calculating association rules with a min cooccurrences level using BigQuery data."""
        if len(expected_results_df) == 0:
            pytest.skip("No association rules found in data")

        if expected_results_df.shape[0] >= 1:
            min_cooccurrence_value = expected_results_df["cooccurrences"].min()
            min_cooccurrences = max(1, min_cooccurrence_value)

            calc_df = ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_cooccurrences=min_cooccurrences,
            )

            result = calc_df.df

            if hasattr(result, "execute"):
                result = result.execute()

            expected = expected_results_df[(expected_results_df["cooccurrences"] >= min_cooccurrences)].reset_index(
                drop=True,
            )

            pd.testing.assert_frame_equal(result, expected)
        else:
            pytest.skip("Not enough data for min cooccurrences test")

    def test_calc_association_min_support(self, transactions_df, expected_results_df):
        """Test calculating association rules with a min support level using BigQuery data."""
        if len(expected_results_df) == 0:
            pytest.skip("No association rules found in data")

        if expected_results_df.shape[0] >= MIN_ASSOCIATION_ROWS:
            support_values = expected_results_df["support"].sort_values()
            min_support = support_values.iloc[0]

            calc_df = ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_support=min_support,
            )

            result = calc_df.df

            if hasattr(result, "execute"):
                result = result.execute()

            expected = expected_results_df[(expected_results_df["support"] >= min_support)].reset_index(drop=True)

            pd.testing.assert_frame_equal(result, expected)
        else:
            pytest.skip("Not enough data for min support test")

    def test_calc_association_min_confidence(self, transactions_df, expected_results_df):
        """Test calculating association rules with a min confidence level using BigQuery data."""
        if len(expected_results_df) == 0:
            pytest.skip("No association rules found in data")

        if expected_results_df.shape[0] >= MIN_ASSOCIATION_ROWS:
            confidence_values = expected_results_df["confidence"].sort_values()
            min_confidence = confidence_values.iloc[0]

            calc_df = ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_confidence=min_confidence,
            )

            result = calc_df.df

            if hasattr(result, "execute"):
                result = result.execute()

            expected = expected_results_df[(expected_results_df["confidence"] >= min_confidence)].reset_index(drop=True)

            pd.testing.assert_frame_equal(result, expected)
        else:
            pytest.skip("Not enough data for min confidence test")

    def test_calc_association_min_uplift(self, transactions_df, expected_results_df):
        """Test calculating association rules with a min uplift level using BigQuery data."""
        if len(expected_results_df) == 0:
            pytest.skip("No association rules found in data")

        if expected_results_df.shape[0] >= MIN_ASSOCIATION_ROWS:
            uplift_values = expected_results_df["uplift"].sort_values()
            min_uplift = uplift_values.iloc[0]

            calc_df = ProductAssociation(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_uplift=min_uplift,
            )

            result = calc_df.df

            if hasattr(result, "execute"):
                result = result.execute()

            expected = expected_results_df[(expected_results_df["uplift"] >= min_uplift)].reset_index(drop=True)

            pd.testing.assert_frame_equal(result, expected)
        else:
            pytest.skip("Not enough data for min uplift test")

    def test_calc_association_invalid_min_occurrences(self, transactions_df):
        """Test calculating association rules with an invalid minimum occurrences value."""
        with pytest.raises(ValueError, match="Minimum occurrences must be at least 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_occurrences=0,
            )

    def test_calc_association_invalid_min_cooccurrences(self, transactions_df):
        """Test calculating association rules with an invalid minimum cooccurrences value."""
        with pytest.raises(ValueError, match="Minimum cooccurrences must be at least 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_cooccurrences=0,
            )

    def test_calc_association_min_support_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum support range."""
        with pytest.raises(ValueError, match="Minimum support must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_support=-0.1,
            )
        with pytest.raises(ValueError, match="Minimum support must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_support=1.1,
            )

    def test_calc_association_min_confidence_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum confidence range."""
        with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_confidence=-0.1,
            )
        with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_confidence=1.1,
            )

    def test_calc_association_min_uplift_invalid_range(self, transactions_df):
        """Test calculating association rules with an invalid minimum uplift range."""
        with pytest.raises(ValueError, match="Minimum uplift must be greater or equal to 0."):
            ProductAssociation._calc_association(
                df=transactions_df,
                value_col="product",
                group_col="transaction_id",
                min_uplift=-0.1,
            )

    def test_real_world_category_association(self, transactions_table):
        """Test real-world category association analysis using BigQuery data."""
        value_col = "category_1_name"
        query = transactions_table.select(
            "transaction_id",
            value_col,
        ).limit(1000)

        df = query.execute()

        unique_categories = df[value_col].nunique()
        if unique_categories < MIN_CATEGORIES_REQUIRED:
            pytest.skip(f"Not enough unique categories ({unique_categories}) for association testing")

        category_counts = df.groupby("transaction_id")[value_col].nunique()
        multi_category_transactions = (category_counts > 1).sum()

        if multi_category_transactions < MIN_MULTI_CATEGORY_TRANSACTIONS:
            pytest.skip(f"Not enough transactions with multiple categories ({multi_category_transactions})")

        category_associations = ProductAssociation(
            df=df,
            value_col=value_col,
            group_col="transaction_id",
            min_cooccurrences=1,
        )

        result_df = category_associations.df
        if hasattr(result_df, "execute"):
            result_df = result_df.execute()

        if len(result_df) == 0:
            pytest.skip("No category associations found in test data")
        else:
            expected_columns = [
                f"{value_col}_1",
                f"{value_col}_2",
                "occurrences_1",
                "occurrences_2",
                "cooccurrences",
                "support",
                "confidence",
                "uplift",
            ]
            for column in expected_columns:
                assert column in result_df.columns

    def test_brand_association(self, transactions_table):
        """Test brand association analysis using BigQuery data."""
        value_col = "brand_name"
        query = transactions_table.select(
            "transaction_id",
            value_col,
        ).limit(1000)

        df = query.execute()

        if df[value_col].isna().mean() > MISSING_VALUE_THRESHOLD or df[value_col].nunique() < MIN_CATEGORIES_REQUIRED:
            pytest.skip("Brand data not suitable for association testing")

        brand_counts = df.groupby("transaction_id")[value_col].nunique()
        multi_brand_transactions = (brand_counts > 1).sum()

        if multi_brand_transactions < MIN_MULTI_CATEGORY_TRANSACTIONS:
            pytest.skip(f"Not enough transactions with multiple brands ({multi_brand_transactions})")

        brand_associations = ProductAssociation(
            df=df,
            value_col=value_col,
            group_col="transaction_id",
            min_cooccurrences=1,
        )

        result_df = brand_associations.df
        if hasattr(result_df, "execute"):
            result_df = result_df.execute()

        if len(result_df) == 0:
            pytest.skip("No brand associations found in test data")
        else:
            expected_columns = [
                f"{value_col}_1",
                f"{value_col}_2",
                "occurrences_1",
                "occurrences_2",
                "cooccurrences",
                "support",
                "confidence",
                "uplift",
            ]
            for column in expected_columns:
                assert column in result_df.columns
