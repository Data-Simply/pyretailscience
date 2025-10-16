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

    def test_group_based_ranking(self):
        """Test basic group-based ranking functionality."""
        electronics_product_1 = 1
        electronics_product_2 = 2
        electronics_product_3 = 3
        apparel_product_1 = 4
        apparel_product_2 = 5
        apparel_product_3 = 6

        rank_1 = 1
        rank_2 = 2
        rank_3 = 3

        df = pd.DataFrame(
            {
                "product_id": [
                    electronics_product_1,
                    electronics_product_2,
                    electronics_product_3,
                    apparel_product_1,
                    apparel_product_2,
                    apparel_product_3,
                ],
                "product_category": ["Electronics", "Electronics", "Electronics", "Apparel", "Apparel", "Apparel"],
                "sales": [1000, 800, 600, 500, 400, 300],
                "margin_pct": [20, 25, 15, 30, 35, 25],
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc"), ("margin_pct", "desc")],
            agg_func="mean",
            group_col="product_category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        # Verify ranks are calculated within groups
        # Electronics group (products 1, 2, 3): sales ranks should be 1, 2, 3
        electronics = result[result["product_category"] == "Electronics"]
        assert (
            electronics.loc[electronics["product_id"] == electronics_product_1, "sales_rank"].iloc[0] == rank_1
        )  # Highest sales
        assert (
            electronics.loc[electronics["product_id"] == electronics_product_2, "sales_rank"].iloc[0] == rank_2
        )  # Middle sales
        assert (
            electronics.loc[electronics["product_id"] == electronics_product_3, "sales_rank"].iloc[0] == rank_3
        )  # Lowest sales

        # Apparel group (products 4, 5, 6): sales ranks should be 1, 2, 3
        apparel = result[result["product_category"] == "Apparel"]
        assert (
            apparel.loc[apparel["product_id"] == apparel_product_1, "sales_rank"].iloc[0] == rank_1
        )  # Highest sales in apparel
        assert (
            apparel.loc[apparel["product_id"] == apparel_product_2, "sales_rank"].iloc[0] == rank_2
        )  # Middle sales in apparel
        assert (
            apparel.loc[apparel["product_id"] == apparel_product_3, "sales_rank"].iloc[0] == rank_3
        )  # Lowest sales in apparel

        # Verify margin ranks within groups
        # Electronics margin ranks: product 2 (25%) = 1, product 1 (20%) = 2, product 3 (15%) = 3
        assert electronics.loc[electronics["product_id"] == electronics_product_2, "margin_pct_rank"].iloc[0] == rank_1
        assert electronics.loc[electronics["product_id"] == electronics_product_1, "margin_pct_rank"].iloc[0] == rank_2
        assert electronics.loc[electronics["product_id"] == electronics_product_3, "margin_pct_rank"].iloc[0] == rank_3

        # Apparel margin ranks: product 5 (35%) = 1, product 4(30%) = 2, product 6 (25%) = 3
        assert apparel.loc[apparel["product_id"] == apparel_product_2, "margin_pct_rank"].iloc[0] == rank_1
        assert apparel.loc[apparel["product_id"] == apparel_product_1, "margin_pct_rank"].iloc[0] == rank_2
        assert apparel.loc[apparel["product_id"] == apparel_product_3, "margin_pct_rank"].iloc[0] == rank_3

    def test_group_vs_global_ranking_difference(self):
        """Test that group-based ranking produces different results than global ranking."""
        product_1_row = 0
        product_2_row = 1
        product_3_row = 2
        product_4_row = 3

        global_rank_1 = 1
        global_rank_2 = 2
        global_rank_3 = 3
        global_rank_4 = 4

        group_rank_1 = 1
        group_rank_2 = 2

        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "category": ["A", "A", "B", "B"],
                "sales": [100, 80, 90, 70],  # Global ranks: 1, 3, 2, 4
            },
        )

        # Global ranking
        global_ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col=None,
        )
        global_result = global_ranker.df.sort_values("product_id").reset_index(drop=True)

        # Group-based ranking
        group_ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc")],
            agg_func="mean",
            group_col="category",
        )
        group_result = group_ranker.df.sort_values("product_id").reset_index(drop=True)

        # Global ranks should be different from group ranks
        # Global: Product 1=1, Product 2=3, Product 3=2, Product 4=4
        assert global_result.loc[product_1_row, "sales_rank"] == global_rank_1
        assert global_result.loc[product_2_row, "sales_rank"] == global_rank_3
        assert global_result.loc[product_3_row, "sales_rank"] == global_rank_2
        assert global_result.loc[product_4_row, "sales_rank"] == global_rank_4

        # Group ranks: Within each category, ranks reset
        # Category A: Product 1=1, Product 2=2
        # Category B: Product 3=1, Product 4=2
        assert group_result.loc[product_1_row, "sales_rank"] == group_rank_1  # Product 1, rank 1 in category A
        assert group_result.loc[product_2_row, "sales_rank"] == group_rank_2  # Product 2, rank 2 in category A
        assert group_result.loc[product_3_row, "sales_rank"] == group_rank_1  # Product 3, rank 1 in category B
        assert group_result.loc[product_4_row, "sales_rank"] == group_rank_2  # Product 4, rank 2 in category B

    def test_invalid_group_col(self, simple_df):
        """Test behavior with an invalid group column."""
        with pytest.raises(ValueError, match="Group column 'non_existent_group' not found"):
            CompositeRank(
                df=simple_df,
                rank_cols=[("spend", "desc")],
                agg_func="mean",
                group_col="non_existent_group",
            )

    @pytest.mark.parametrize(
        ("invalid_group_col", "expected_error_message"),
        [
            (123, "Group column '123' not found"),
            (12.5, "Group column '12.5' not found"),
            (True, "Group column 'True' not found"),
            (["invalid"], r"Group column '\['invalid'\]' not found"),
            ({"key": "value"}, r"Group column '\{'key': 'value'\}' not found"),
        ],
    )
    def test_invalid_group_col_type(self, simple_df, invalid_group_col, expected_error_message):
        """Test behavior with invalid group_col type."""
        with pytest.raises(ValueError, match=expected_error_message):
            CompositeRank(
                df=simple_df,
                rank_cols=[("spend", "desc")],
                agg_func="mean",
                group_col=invalid_group_col,
            )

    def test_group_col_with_null_values(self):
        """Test behavior when group column contains NULL values."""
        # Test data constants
        expected_null_group_size = 2
        expected_electronics_group_size = 2
        expected_apparel_group_size = 1
        expected_rank_1 = 1
        expected_rank_2 = 2

        # Test data with None values in group column (common in retail data)
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "category": ["Electronics", None, "Apparel", None, "Electronics"],
                "sales": [100, 80, 90, 70, 60],
                "margin": [20, 15, 25, 10, 18],
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc"), ("margin", "desc")],
            agg_func="mean",
            group_col="category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        # Verify NULL values form their own ranking group
        null_rows = result[result["category"].isna()]
        assert len(null_rows) == expected_null_group_size

        # NULL group: sales [80, 70] -> ranks [1, 2], margin [15, 10] -> ranks [1, 2]
        null_sales_ranks = null_rows.sort_values("product_id")["sales_rank"].tolist()
        null_margin_ranks = null_rows.sort_values("product_id")["margin_rank"].tolist()
        assert null_sales_ranks == [expected_rank_1, expected_rank_2]  # Product 2 (80) > Product 4 (70)
        assert null_margin_ranks == [expected_rank_1, expected_rank_2]  # Product 2 (15) > Product 4 (10)

        # Verify Electronics group ranks independently
        electronics_rows = result[result["category"] == "Electronics"]
        assert len(electronics_rows) == expected_electronics_group_size

        # Electronics: sales [100, 60] -> ranks [1, 2], margin [20, 18] -> ranks [1, 2]
        elec_sales_ranks = electronics_rows.sort_values("product_id")["sales_rank"].tolist()
        elec_margin_ranks = electronics_rows.sort_values("product_id")["margin_rank"].tolist()
        assert elec_sales_ranks == [expected_rank_1, expected_rank_2]  # Product 1 (100) > Product 5 (60)
        assert elec_margin_ranks == [expected_rank_1, expected_rank_2]  # Product 1 (20) > Product 5 (18)

        # Verify Apparel group (single item gets rank 1)
        apparel_rows = result[result["category"] == "Apparel"]
        assert len(apparel_rows) == expected_apparel_group_size
        assert apparel_rows["sales_rank"].iloc[0] == expected_rank_1
        assert apparel_rows["margin_rank"].iloc[0] == expected_rank_1

    def test_group_with_ignore_ties(self):
        """Test group-based ranking with ignore_ties=True."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "group": ["A", "A", "B", "B"],
                "value": [100, 100, 50, 50],  # Ties within each group
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("value", "desc")],
            agg_func="mean",
            group_col="group",
            ignore_ties=True,
        )

        result = ranker.df.sort_values("id").reset_index(drop=True)

        # Within each group, ranks should be unique even with ties
        group_a = result[result["group"] == "A"]["value_rank"].tolist()
        group_b = result[result["group"] == "B"]["value_rank"].tolist()

        # Each group should have unique ranks (1, 2) despite tied values
        assert set(group_a) == {1, 2}
        assert set(group_b) == {1, 2}

    @pytest.mark.parametrize(
        ("agg_func", "expected_electronics", "expected_apparel"),
        [
            ("mean", {"1": 1.5, "2": 1.5, "3": 3.0}, {"4": 1.5, "5": 1.5, "6": 3.0}),
            ("sum", {"1": 3, "2": 3, "3": 6}, {"4": 3, "5": 3, "6": 6}),
            ("min", {"1": 1, "2": 1, "3": 3}, {"4": 1, "5": 1, "6": 3}),
            ("max", {"1": 2, "2": 2, "3": 3}, {"4": 2, "5": 2, "6": 3}),
        ],
    )
    def test_group_agg_functions(self, agg_func, expected_electronics, expected_apparel):
        """Test different aggregation functions with group-based ranking."""
        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5, 6],
                "category": ["Electronics", "Electronics", "Electronics", "Apparel", "Apparel", "Apparel"],
                "sales": [1000, 800, 600, 500, 400, 300],  # Within group ranks: E=[1,2,3], A=[1,2,3]
                "margin": [20, 25, 15, 30, 35, 25],  # Within group ranks: E=[2,1,3], A=[2,1,3]
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("sales", "desc"), ("margin", "desc")],
            agg_func=agg_func,
            group_col="category",
        )

        result = ranker.df.sort_values("product_id").reset_index(drop=True)

        # Test Electronics category
        for product_id, expected in expected_electronics.items():
            actual = result.loc[
                result["product_id"] == int(product_id),
                "composite_rank",
            ].iloc[0]
            assert actual == pytest.approx(expected, rel=1e-3), (
                f"{agg_func} for Electronics product {product_id} failed"
            )

        # Test Apparel category
        for product_id, expected in expected_apparel.items():
            actual = result.loc[
                result["product_id"] == int(product_id),
                "composite_rank",
            ].iloc[0]
            assert actual == pytest.approx(expected, rel=1e-3), f"{agg_func} for Apparel product {product_id} failed"

    def test_single_group(self):
        """Test behavior when all data belongs to a single group."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "category": ["A", "A", "A"],  # All same group
                "value": [100, 200, 150],
            },
        )

        ranker = CompositeRank(
            df=df,
            rank_cols=[("value", "desc")],
            agg_func="mean",
            group_col="category",
        )

        result = ranker.df.sort_values("id").reset_index(drop=True)

        # Should rank normally within the single group
        expected_ranks = [3, 1, 2]  # Based on values [100, 200, 150] descending
        actual_ranks = result["value_rank"].tolist()
        assert actual_ranks == expected_ranks

    def test_group_with_ibis_table(self):
        """Test that group-based ranking works with Ibis table input."""
        id_1_row = 0
        id_2_row = 1
        id_3_row = 2
        id_4_row = 3

        rank_1_in_group = 1
        rank_2_in_group = 2

        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "group": ["A", "A", "B", "B"],
                "value": [100, 80, 90, 70],
            },
        )

        ibis_table = ibis.memtable(df)

        ranker = CompositeRank(
            df=ibis_table,
            rank_cols=[("value", "desc")],
            agg_func="mean",
            group_col="group",
        )

        result = ranker.df.sort_values("id").reset_index(drop=True)

        # Verify grouped ranking worked
        assert result.loc[id_1_row, "value_rank"] == rank_1_in_group  # id=1, rank 1 in group A
        assert result.loc[id_2_row, "value_rank"] == rank_2_in_group  # id=2, rank 2 in group A
        assert result.loc[id_3_row, "value_rank"] == rank_1_in_group  # id=3, rank 1 in group B
        assert result.loc[id_4_row, "value_rank"] == rank_2_in_group  # id=4, rank 2 in group B
