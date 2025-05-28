"""Tests for the RFMSegmentation class."""

import ibis
import pandas as pd
import pytest
from freezegun import freeze_time

from pyretailscience.options import ColumnHelper
from pyretailscience.segmentation.rfm import RFMSegmentation

cols = ColumnHelper()


class TestRFMSegmentation:
    """Tests for the RFMSegmentation class."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                cols.transaction_id: [101, 102, 103, 104, 105],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_date: [
                    "2025-03-01",
                    "2025-02-15",
                    "2025-01-30",
                    "2025-03-10",
                    "2025-02-20",
                ],
            },
        )

    @pytest.fixture
    def expected_df(self):
        """Returns the expected DataFrame for testing segmentation."""
        return pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "frequency": [1, 1, 1, 1, 1],
                "monetary": [100.0, 200.0, 150.0, 300.0, 250.0],
                "r_score": [3, 1, 0, 4, 2],
                "f_score": [0, 1, 2, 3, 4],
                "m_score": [0, 2, 1, 4, 3],
                "rfm_segment": [300, 112, 21, 434, 243],
                "fm_segment": [0, 12, 21, 34, 43],
            },
        ).set_index("customer_id")

    @pytest.fixture
    def larger_df(self):
        """Return a larger DataFrame for testing custom segments."""
        return pd.DataFrame(
            {
                cols.customer_id: list(range(1, 21)),
                cols.transaction_id: list(range(101, 121)),
                cols.unit_spend: [100.0 + i * 10 for i in range(20)],
                cols.transaction_date: ["2025-03-01"] * 20,
            },
        )

    def test_correct_rfm_segmentation(self, base_df, expected_df):
        """Test that the RFM segmentation correctly calculates the RFM scores and segments."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)
        result_df = rfm_segmentation.df
        expected_df["recency_days"] = [16, 30, 46, 7, 25]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_df["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    def test_handles_dataframe_with_missing_columns(self):
        """Test that the method raises an error when required columns are missing."""
        base_df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 150.0],
                cols.transaction_id: [101, 102, 103],
            },
        )

        with pytest.raises(ValueError):
            RFMSegmentation(df=base_df, current_date="2025-03-17")

    def test_single_customer(self):
        """Test that the method correctly calculates RFM segmentation for a single customer."""
        df_single_customer = pd.DataFrame(
            {
                cols.customer_id: [1],
                cols.transaction_id: [101],
                cols.unit_spend: [200.0],
                cols.transaction_date: ["2025-03-01"],
            },
        )
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=df_single_customer, current_date=current_date)
        result_df = rfm_segmentation.df
        assert result_df.loc[1, "rfm_segment"] == 0

    def test_multiple_transactions_per_customer(self):
        """Test that the method correctly handles multiple transactions for the same customer."""
        df_multiple_transactions = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 1, 1],
                cols.transaction_id: [101, 102, 103, 104, 105],
                cols.unit_spend: [120.0, 250.0, 180.0, 300.0, 220.0],
                cols.transaction_date: [
                    "2025-03-01",
                    "2025-02-15",
                    "2025-01-10",
                    "2025-03-10",
                    "2025-02-25",
                ],
            },
        )
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=df_multiple_transactions, current_date=current_date)
        result_df = rfm_segmentation.df

        assert result_df.loc[1, "rfm_segment"] == 0

    def test_calculates_rfm_correctly_for_all_customers(self, base_df):
        """Test that RFM scores are calculated correctly for all customers."""
        current_date = "2025-03-17"
        expected_customer_count = 5
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)
        result_df = rfm_segmentation.df

        assert len(result_df) == expected_customer_count
        assert "rfm_segment" in result_df.columns

    @freeze_time("2025-03-19")
    def test_rfm_segmentation_with_no_date(self, base_df, expected_df):
        """Test that the RFM segmentation correctly calculates the RFM scores and segments."""
        rfm_segmentation = RFMSegmentation(df=base_df)
        result_df = rfm_segmentation.df
        expected_df["recency_days"] = [18, 32, 48, 9, 27]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_df["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    def test_invalid_current_date_type(self, base_df):
        """Test that RFMSegmentation raises a TypeError when an invalid current_date is provided."""
        with pytest.raises(
            TypeError,
            match="current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None",
        ):
            RFMSegmentation(base_df, current_date=12345)

    def test_invalid_df_type(self):
        """Test that RFMSegmentation raises a TypeError when df is neither a DataFrame nor an Ibis Table."""
        invalid_df = "this is not a dataframe"

        with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an Ibis Table"):
            RFMSegmentation(df=invalid_df, current_date="2025-03-17")

    def test_custom_bins_with_integers(self, larger_df):
        """Test that custom number of bins works correctly."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=5,
            f_segments=3,
            m_segments=4,
        )
        result_df = rfm_segmentation.df
        max_r, max_f, max_m = 4, 2, 3

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m

        assert result_df["r_score"].min() == 0
        assert result_df["f_score"].min() == 0
        assert result_df["m_score"].min() == 0

    def test_custom_cut_points(self, larger_df):
        """Test that custom cut points work correctly."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=[0.2, 0.4, 0.6, 0.8],
            f_segments=[0.33, 0.66],
            m_segments=[0.25, 0.5, 0.75],
        )
        result_df = rfm_segmentation.df

        max_r, max_f, max_m = 4, 2, 3

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m
        assert result_df["r_score"].min() == 0
        assert result_df["f_score"].min() == 0
        assert result_df["m_score"].min() == 0

    def test_compatibility_default_values(self, base_df, expected_df):
        """Test that default behavior (10 bins) works as before."""
        current_date = "2025-03-17"

        rfm_segmentation_implicit = RFMSegmentation(
            df=base_df,
            current_date=current_date,
        )

        result_implicit = rfm_segmentation_implicit.df

        expected_df["recency_days"] = [16, 30, 46, 7, 25]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_implicit["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_implicit.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    def test_validation_integer_segments_out_of_range(self, base_df):
        """Test validation for integer segments out of range."""
        with pytest.raises(ValueError, match="r_segments must be between 1 and 10 when specified as an integer"):
            RFMSegmentation(base_df, r_segments=0)

        with pytest.raises(ValueError, match="f_segments must be between 1 and 10 when specified as an integer"):
            RFMSegmentation(base_df, f_segments=11)

        with pytest.raises(ValueError, match="m_segments must be between 1 and 10 when specified as an integer"):
            RFMSegmentation(base_df, m_segments=-1)

    def test_validation_empty_cut_points(self, base_df):
        """Test validation for empty cut points list."""
        with pytest.raises(ValueError, match="r_segments must contain between 1 and 9 cut points"):
            RFMSegmentation(base_df, r_segments=[])

    def test_validation_too_many_cut_points(self, base_df):
        """Test validation for too many cut points."""
        with pytest.raises(ValueError, match="f_segments must contain between 1 and 9 cut points"):
            RFMSegmentation(base_df, f_segments=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    def test_validation_cut_points_out_of_range(self, base_df):
        """Test validation for cut points outside [0, 1] range."""
        with pytest.raises(ValueError, match="All cut points in r_segments must be between 0 and 1"):
            RFMSegmentation(base_df, r_segments=[0.5, 1.5])

        with pytest.raises(ValueError, match="All cut points in f_segments must be between 0 and 1"):
            RFMSegmentation(base_df, f_segments=[-0.1, 0.5])

    def test_validation_non_numeric_cut_points(self, base_df):
        """Test validation for non-numeric cut points."""
        with pytest.raises(ValueError, match="All cut points in m_segments must be numeric"):
            RFMSegmentation(base_df, m_segments=[0.5, "invalid"])

    def test_validation_duplicate_cut_points(self, base_df):
        """Test validation for duplicate cut points."""
        with pytest.raises(ValueError, match="Cut points in r_segments must be unique"):
            RFMSegmentation(base_df, r_segments=[0.3, 0.5, 0.3])

    def test_validation_unsorted_cut_points(self, base_df):
        """Test validation for unsorted cut points."""
        with pytest.raises(ValueError, match="Cut points in f_segments must be in ascending order"):
            RFMSegmentation(base_df, f_segments=[0.7, 0.3, 0.9])

    def test_validation_invalid_segment_type(self, base_df):
        """Test validation for invalid segment parameter type."""
        with pytest.raises(TypeError, match="m_segments must be an integer or a list of floats"):
            RFMSegmentation(base_df, m_segments="invalid")

    def test_mixed_segment_types(self, larger_df):
        """Test using different segment types for different metrics."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=5,  # Integer
            f_segments=[0.5],  # Cut points
            m_segments=[0.25, 0.5, 0.75],  # Cut points
        )
        result_df = rfm_segmentation.df
        max_r, max_f, max_m, expected_length = 4, 1, 3, 20

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m
        assert len(result_df) == expected_length

    def test_single_bin_segments(self, larger_df):
        """Test using single bins for all segments."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=1,
            f_segments=1,
            m_segments=1,
        )
        result_df = rfm_segmentation.df

        assert (result_df["r_score"] == 0).all()
        assert (result_df["f_score"] == 0).all()
        assert (result_df["m_score"] == 0).all()
        assert (result_df["rfm_segment"] == 0).all()
        assert (result_df["fm_segment"] == 0).all()

    def test_edge_case_cut_points(self, larger_df):
        """Test edge cases for cut points (0.0 and 1.0)."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=[0.0, 1.0],
            f_segments=[1.0],
            m_segments=[0.0, 0.5],
        )
        result_df = rfm_segmentation.df
        max_r, max_f, max_m, expected_length = 1, 0, 2, 20

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m
        assert len(result_df) == expected_length

    def test_lazy_execution_with_ibis_table(self, base_df):
        """Test lazy execution of Ibis table on accessing .df."""
        ibis_table = ibis.memtable(base_df)
        rfm = RFMSegmentation(df=ibis_table, current_date="2025-03-17")

        result_df = rfm.df
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
