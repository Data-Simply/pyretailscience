"""Tests for the CompositeRank class."""

import ibis
import pandas as pd
import pytest

from pyretailscience.analysis.composite_rank import CompositeRank


class TestCompositeRank:
    """Tests for the CompositeRank class."""

    @pytest.fixture
    def simple_df(self):
        """Create a simple DataFrame fixture for basic tests."""
        return pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "spend": [100, 150, 75],
                "customers": [20, 30, 15],
            },
        )

    def test_basic_functionality(self):
        """Test the basic functionality of the CompositeRank class."""
        # Create test data
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],
                "customers": [20, 30, 15, 40, 25],
                "spend_per_customer": [5.0, 5.0, 5.0, 5.0, 5.0],
            },
        )

        # Test with ascending and descending columns
        rank_cols = [
            ("spend", "desc"),  # Higher spend is better
            ("customers", "desc"),  # Higher customer count is better
            ("spend_per_customer", "desc"),  # Higher spend per customer is better
        ]

        # Create expected dataframe with ranks
        expected_df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],
                "customers": [20, 30, 15, 40, 25],
                "spend_per_customer": [5.0, 5.0, 5.0, 5.0, 5.0],
                "spend_rank": [4, 2, 5, 1, 3],
                "customers_rank": [4, 2, 5, 1, 3],
                "spend_per_customer_rank": [1, 1, 1, 1, 1],
                "composite_rank": [3.0, 1.6666667, 3.6666667, 1.0, 2.3333333],
            },
        )

        # Create instance with mean aggregation
        cr = CompositeRank(df=df, rank_cols=rank_cols, agg_func="mean", ignore_ties=False)

        # Extract only the relevant columns for comparison
        result_subset = cr.df.sort_values("product_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(result_subset, expected_df)

    def test_string_column_specs(self, simple_df):
        """Test that the class accepts string column specifications."""
        # Create expected dataframe for ascending rank
        expected_df = pd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "spend_rank": [2, 3, 1],  # Ascending: lowest values (75) get rank 1
                "customers_rank": [2, 3, 1],  # Ascending: lowest values (15) get rank 1
            },
        )

        # Test with just column names (should default to ascending)
        cr = CompositeRank(df=simple_df, rank_cols=["spend", "customers"], agg_func="mean", ignore_ties=False)

        # Sort both dataframes for comparison
        result_subset = cr.df.sort_values("product_id").reset_index(drop=True)[
            ["product_id", "spend_rank", "customers_rank"]
        ]

        pd.testing.assert_frame_equal(result_subset, expected_df)

    def test_ignore_ties(self):
        """Test the behavior with the ignore_ties parameter."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "spend": [100, 100, 150, 150],
            },
        )

        # Without ignore_ties (should have ties)
        cr_with_ties = CompositeRank(df=df, rank_cols=[("spend", "desc")], agg_func="mean", ignore_ties=False)

        result_with_ties = cr_with_ties.df
        # Create expected dataframe for tie ranks (descending)
        expected_df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "spend_rank": [3, 3, 1, 1],  # Descending: [100, 100] gets rank 3, [150, 150] gets rank 1
            },
        )

        # Sort both dataframes for comparison
        result_subset = result_with_ties.sort_values("product_id").reset_index(drop=True)[["product_id", "spend_rank"]]

        pd.testing.assert_frame_equal(result_subset, expected_df)

        # With ignore_ties (should have unique ranks)
        cr_no_ties = CompositeRank(df=df, rank_cols=[("spend", "desc")], agg_func="mean", ignore_ties=True)

        # Verify all ranks are unique
        expected_unique_ranks = 4
        assert len(cr_no_ties.df["spend_rank"].unique()) == expected_unique_ranks

    @pytest.mark.parametrize(
        ("agg_func", "expected_values"),
        [
            ("mean", {"1": 3.0, "2": 3.6667, "3": 3.3333, "4": 2.3333, "5": 2.6667}),
            ("sum", {"1": 9, "2": 11, "3": 10, "4": 7, "5": 8}),
            ("min", {"1": 2, "2": 2, "3": 1, "4": 1, "5": 2}),
            ("max", {"1": 4, "2": 5, "3": 5, "4": 5, "5": 3}),
        ],
    )
    def test_different_agg_functions(self, agg_func, expected_values):
        """Test different aggregation functions using parametrization."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "spend": [100, 150, 75, 200, 125],  # Ranks descending: 4, 2, 5, 1, 3
                "customers": [30, 10, 20, 40, 25],  # Ranks descending: 2, 5, 4, 1, 3
                "profit": [50, 30, 70, 10, 60],  # Ranks descending: 3, 4, 1, 5, 2
            },
        )
        rank_cols = [("spend", "desc"), ("customers", "desc"), ("profit", "desc")]

        # Create instance with specified aggregation function
        cr = CompositeRank(df=df, rank_cols=rank_cols, agg_func=agg_func, ignore_ties=False)

        # Verify composite rank for each product
        for product_id, expected in expected_values.items():
            product_id_int = int(product_id)
            actual = cr.df.loc[cr.df["product_id"] == product_id_int, "composite_rank"].iloc[0]
            assert actual == pytest.approx(expected, rel=1e-3), f"{agg_func} for product {product_id} failed"

    def test_invalid_column(self, simple_df):
        """Test behavior with an invalid column."""
        rank_cols = [("spend", "desc"), ("non_existent_column", "asc")]

        with pytest.raises(ValueError, match="Column 'non_existent_column' not found"):
            CompositeRank(df=simple_df, rank_cols=rank_cols, agg_func="mean", ignore_ties=False)

    def test_invalid_sort_order(self, simple_df):
        """Test behavior with an invalid sort order."""
        rank_cols = [("spend", "invalid_sort_order")]

        with pytest.raises(ValueError, match="Sort order must be one of"):
            CompositeRank(df=simple_df, rank_cols=rank_cols, agg_func="mean", ignore_ties=False)

    def test_invalid_column_specification(self, simple_df):
        """Test behavior with an invalid column specification."""
        # Test with a tuple that has more than 2 elements
        rank_cols = [("spend", "desc", "extra_value")]

        with pytest.raises(ValueError, match="Column specification must be a string or a tuple of"):
            CompositeRank(df=simple_df, rank_cols=rank_cols, agg_func="mean", ignore_ties=False)

    def test_invalid_agg_func(self, simple_df):
        """Test behavior with an invalid aggregation function."""
        with pytest.raises(ValueError, match="Aggregation function must be one of"):
            CompositeRank(df=simple_df, rank_cols=[("spend", "desc")], agg_func="invalid_func", ignore_ties=False)

    def test_ibis_table_input(self, simple_df):
        """Test that the class works with Ibis table input."""
        # Convert to Ibis table
        ibis_table = ibis.memtable(simple_df)

        # Create CompositeRank instance
        cr = CompositeRank(
            df=ibis_table,
            rank_cols=[("spend", "desc"), ("customers", "desc")],
            agg_func="mean",
            ignore_ties=False,
        )

        assert all(col in cr.df.columns for col in ["spend_rank", "customers_rank", "composite_rank"])

    def test_df_cache(self, simple_df):
        """Test that the df property caches results."""
        # Create instance
        cr = CompositeRank(df=simple_df, rank_cols=[("spend", "desc")], agg_func="mean", ignore_ties=False)

        # Access df property to populate cache
        result1 = cr.df

        # Verify _df is populated
        assert cr._df is not None

        # Access df property again to use cache
        result2 = cr.df

        # Verify both results are the same object (cached)
        assert result1 is result2
