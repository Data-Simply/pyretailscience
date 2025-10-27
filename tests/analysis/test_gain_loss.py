"""Tests for the GainLoss class in the gain_loss module."""

import datetime

import ibis
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyretailscience.analysis.gain_loss import GainLoss
from pyretailscience.options import option_context


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.mark.parametrize(
    ("focus_p1", "comparison_p1", "focus_p2", "comparison_p2", "focus_diff", "comparison_diff", "expected"),
    [
        # Test when a new customer is added (focus_p1 and comparison_p1 are 0)
        (0, 0, 100, 0, 100, 0, (100, 0, 0, 0, 0, 0)),
        # Test when a customer is lost (focus_p2 and comparison_p2 are 0)
        (100, 0, 0, 0, -100, 0, (0, -100, 0, 0, 0, 0)),
        # Test when both focus and comparison increase, but focus increase is greater
        (100, 100, 150, 120, 50, 20, (0, 0, 50, 0, 0, 0)),
        # Test when both focus and comparison increase, but comparison increase is greater
        (100, 100, 120, 150, 20, 50, (0, 0, 20, 0, 0, 0)),
        # Test when focus increases and comparison decreases, with focus change greater
        (100, 100, 150, 80, 50, -20, (0, 0, 30, 0, 20, 0)),
        # Test when focus increases and comparison decreases, with comparison change greater
        (100, 100, 110, 50, 10, -50, (0, 0, 0, 0, 10, 0)),
        # Test when both focus and comparison decrease, but focus decrease is greater
        (100, 100, 50, 80, -50, -20, (0, 0, 0, -50, 0, 0)),
        # Test when both focus and comparison decrease, but comparison decrease is greater
        (100, 100, 80, 50, -20, -50, (0, 0, 0, -20, 0, 0)),
        # Test when focus decreases and comparison increases, with focus change greater
        (100, 100, 50, 110, -50, 10, (0, 0, 0, -40, 0, -10)),
        # Test when focus decreases and comparison increases, with comparison change greater
        (100, 100, 80, 150, -20, 50, (0, 0, 0, 0, 0, -20)),
        # Test switch from comparison to focus, with focus change greater
        (100, 100, 180, 50, 80, -50, (0, 0, 30, 0, 50, 0)),
        # Test switch from comparison to focus, with comparison change greater
        (100, 100, 150, 20, 50, -80, (0, 0, 0, 0, 50, 0)),
        # Test switch from focus to comparison, with focus change greater
        (100, 100, 20, 150, -80, 50, (0, 0, 0, -30, 0, -50)),
        # Test switch from focus to comparison, with comparison change greater
        (100, 100, 50, 180, -50, 80, (0, 0, 0, 0, 0, -50)),
        # Test when there's no change in focus or comparison
        (100, 100, 100, 100, 0, 0, (0, 0, 0, 0, 0, 0)),
        # Test when a new customer is added (focus_p1 and comparison_p1 are 0)
        (0, 0, 1, 0, 1, 0, (1, 0, 0, 0, 0, 0)),
        # Test when a customer is lost (focus_p2 and comparison_p2 are 0)
        (1, 0, 0, 0, -1, 0, (0, -1, 0, 0, 0, 0)),
        # Test when both focus and comparison increase
        (0, 0, 1, 1, 1, 1, (1, 0, 0, 0, 0, 0)),
        # Test when focus increases and comparison decreases
        (0, 1, 1, 0, 1, -1, (0, 0, 0, 0, 1, 0)),
        # Test when both focus and comparison decrease
        (1, 1, 0, 0, -1, -1, (0, -1, 0, 0, 0, 0)),
        # Test when focus decreases and comparison increases
        (1, 0, 0, 1, -1, 1, (0, 0, 0, 0, 0, -1)),
        # Test when there's no change in focus or comparison
        (1, 1, 1, 1, 0, 0, (0, 0, 0, 0, 0, 0)),
    ],
)
def test_process_customer_group(
    focus_p1,
    comparison_p1,
    focus_p2,
    comparison_p2,
    focus_diff,
    comparison_diff,
    expected,
):
    """Test the process_customer_group method of the GainLoss class."""
    result = GainLoss.process_customer_group(
        focus_p1=focus_p1,
        comparison_p1=comparison_p1,
        focus_p2=focus_p2,
        comparison_p2=comparison_p2,
        focus_diff=focus_diff,
        comparison_diff=comparison_diff,
    )
    assert result == expected


@pytest.fixture
def sample_df():
    """Sample transaction DataFrame for testing."""
    data = {
        "transaction_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "customer_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4],
        "transaction_date": [
            datetime.date(2023, 1, 15),
            datetime.date(2023, 1, 20),
            datetime.date(2023, 2, 5),
            datetime.date(2023, 2, 10),
            datetime.date(2023, 3, 1),
            datetime.date(2023, 3, 15),
            datetime.date(2023, 3, 20),
            datetime.date(2023, 4, 10),
            datetime.date(2023, 4, 25),
            datetime.date(2023, 5, 5),
            datetime.date(2023, 5, 20),
            datetime.date(2023, 6, 10),
        ],
        "unit_spend": [100, 150, 200, 120, 160, 210, 130, 170, 220, 140, 180, 230],
        "brand": [
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
        ],
    }
    return pd.DataFrame(data)


def test_overlap_in_p1_p2_raises(sample_df):
    """Test that GainLoss raises a ValueError when p1 and p2 indices overlap."""
    p1 = [True] * 6 + [False] * 6
    p2 = [True] + [False] * 11

    with pytest.raises(ValueError, match="p1_index and p2_index should not overlap"):
        GainLoss(
            df=sample_df,
            p1_index=p1,
            p2_index=p2,
            focus_group_index=[True] * 6 + [False] * 6,
            focus_group_name="Brand A",
            comparison_group_index=[False] * 6 + [True] * 6,
            comparison_group_name="Brand B",
        )


def test_overlap_in_focus_comparison_raises(sample_df):
    """Test that GainLoss raises a ValueError when focus and comparison indices overlap."""
    with pytest.raises(ValueError, match="focus_group_index and comparison_group_index should not overlap"):
        GainLoss(
            df=sample_df,
            p1_index=[True] * 6 + [False] * 6,
            p2_index=[False] * 6 + [True] * 6,
            focus_group_index=[True] * 6 + [False] * 6,
            focus_group_name="Brand A",
            comparison_group_index=[True] * 6 + [False] * 6,
            comparison_group_name="Brand B",
        )


def test_missing_required_column_raises():
    """Test that GainLoss raises a ValueError if required columns are missing in the DataFrame."""
    df = pd.DataFrame(
        {
            "spend": [100, 200],
            "group": ["A", "B"],
        },
    )
    with pytest.raises(ValueError, match="columns are required but missing"):
        GainLoss(
            df=df,
            p1_index=[True, False],
            p2_index=[False, True],
            focus_group_index=[True, False],
            focus_group_name="A",
            comparison_group_index=[False, True],
            comparison_group_name="B",
        )


def test_valid_gainloss(sample_df):
    """Test that a GainLoss object is successfully with valid input."""
    gl = GainLoss(
        df=sample_df,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )

    assert isinstance(gl, GainLoss)


def test_calc_gains_loss_table_without_group():
    """Test _calc_gains_loss_table with a flat DataFrame (no group index)."""
    df = pd.DataFrame(
        {
            "focus_p1": [10, -5, 20],
            "comparison_p1": [5, 2, 1],
            "total_p1": [15, -3, 21],
            "focus_p2": [12, -3, 25],
            "comparison_p2": [8, 5, 3],
            "total_p2": [20, 2, 28],
            "focus_diff": [2, 2, 5],
            "comparison_diff": [3, 3, 2],
            "total_diff": [5, 5, 7],
            "new": [1, 0, 2],
            "lost": [0, -1, 0],
            "increased_focus": [1, 1, 3],
            "decreased_focus": [0, 0, 0],
            "switch_from_comparison": [0, 1, 0],
            "switch_to_comparison": [0, 0, 0],
        },
    )
    ibis_table = ibis.memtable(df)
    result_ibis = GainLoss._calc_gains_loss_table(ibis_table)
    result = result_ibis.execute()
    expected = pd.DataFrame(
        {
            "focus_p1": [25],
            "comparison_p1": [8],
            "total_p1": [33],
            "focus_p2": [34],
            "comparison_p2": [16],
            "total_p2": [50],
            "focus_diff": [9],
            "comparison_diff": [8],
            "total_diff": [17],
            "new": [3],
            "lost": [-1],
            "increased_focus": [5],
            "decreased_focus": [0],
            "switch_from_comparison": [1],
            "switch_to_comparison": [0],
        },
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_calc_gains_loss_table_with_group():
    """Test _calc_gains_loss_table with grouped DataFrame."""
    df = pd.DataFrame(
        {
            "group": ["Group A", "Group A", "Group B"],
            "id": [1, 2, 1],
            "focus_p1": [10, 15, 5],
            "comparison_p1": [1, 2, 3],
            "total_p1": [11, 17, 8],
            "focus_p2": [12, 18, 8],
            "comparison_p2": [3, 5, 6],
            "total_p2": [15, 23, 14],
            "focus_diff": [2, 3, 3],
            "comparison_diff": [2, 3, 3],
            "total_diff": [4, 6, 6],
            "new": [1, 1, 1],
            "lost": [0, 0, 0],
            "increased_focus": [1, 2, 2],
            "decreased_focus": [0, 0, 0],
            "switch_from_comparison": [0, 0, 0],
            "switch_to_comparison": [0, 0, 0],
        },
    )
    ibis_table = ibis.memtable(df)
    result_ibis = GainLoss._calc_gains_loss_table(ibis_table, group_col="group")
    result = result_ibis.execute().set_index("group").sort_index()

    expected = pd.DataFrame(
        {
            "focus_p1": [25, 5],
            "comparison_p1": [3, 3],
            "total_p1": [28, 8],
            "focus_p2": [30, 8],
            "comparison_p2": [8, 6],
            "total_p2": [38, 14],
            "focus_diff": [5, 3],
            "comparison_diff": [5, 3],
            "total_diff": [10, 6],
            "new": [2, 1],
            "lost": [0, 0],
            "increased_focus": [3, 2],
            "decreased_focus": [0, 0],
            "switch_from_comparison": [0, 0],
            "switch_to_comparison": [0, 0],
        },
        index=pd.Index(["Group A", "Group B"], name="group"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_plot_returns_figure_from_gainloss(sample_df):
    """Test that the `plot` method of the GainLoss class returns a valid matplotlib Axes object."""
    p1 = sample_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2 = sample_df["transaction_date"] >= datetime.date(2023, 5, 1)

    focus_group_index = sample_df["brand"] == "Brand A"
    comparison_group_index = sample_df["brand"] == "Brand B"

    gl = GainLoss(
        df=sample_df,
        p1_index=p1,
        p2_index=p2,
        focus_group_index=focus_group_index,
        focus_group_name="Brand A",
        comparison_group_index=comparison_group_index,
        comparison_group_name="Brand B",
    )

    fig = gl.plot()
    assert isinstance(fig, Axes)


def test_with_custom_column_names(sample_df):
    """Test GainLoss with custom column names."""
    rename_mapping = {
        "customer_id": "cust_identifier",
        "unit_spend": "total_revenue",
    }
    custom_df = sample_df.rename(columns=rename_mapping)
    p1_index = custom_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2_index = custom_df["transaction_date"] >= datetime.date(2023, 5, 1)
    focus_group_index = custom_df["brand"] == "Brand A"
    comparison_group_index = custom_df["brand"] == "Brand B"

    with option_context("column.customer_id", "cust_identifier", "column.unit_spend", "total_revenue"):
        gl = GainLoss(
            df=custom_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name="Brand A",
            comparison_group_index=comparison_group_index,
            comparison_group_name="Brand B",
            value_col="total_revenue",
        )
        gain_loss_df = gl.gain_loss_df
        assert isinstance(gain_loss_df, pd.DataFrame)
        assert not gain_loss_df.empty


def test_gainloss_with_ibis_table_input(sample_df):
    """Test GainLoss with direct Ibis table input (new capability from ticket #152)."""
    ibis_table = ibis.memtable(sample_df)

    p1_index = sample_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2_index = sample_df["transaction_date"] >= datetime.date(2023, 5, 1)
    focus_group_index = sample_df["brand"] == "Brand A"
    comparison_group_index = sample_df["brand"] == "Brand B"

    gl = GainLoss(
        df=ibis_table,
        p1_index=p1_index,
        p2_index=p2_index,
        focus_group_index=focus_group_index,
        focus_group_name="Brand A",
        comparison_group_index=comparison_group_index,
        comparison_group_name="Brand B",
    )

    assert isinstance(gl, GainLoss)
    assert isinstance(gl.table, ibis.Table)
    assert isinstance(gl.df, pd.DataFrame)


def test_type_validation_with_invalid_input():
    """Test type validation for invalid inputs (enhanced for Ibis)."""
    with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an ibis Table"):
        GainLoss(
            df="invalid_type",  # String instead of DataFrame/Table
            p1_index=[True, False],
            p2_index=[False, True],
            focus_group_index=[True, False],
            focus_group_name="A",
            comparison_group_index=[False, True],
            comparison_group_name="B",
        )


def test_mismatched_index_lengths_raises(sample_df):
    """Test that GainLoss raises ValueError when index lengths don't match."""
    with pytest.raises(ValueError, match="should have the same length"):
        GainLoss(
            df=sample_df,
            p1_index=[True, False],  # Length 2
            p2_index=[False, True, False],  # Length 3 - mismatch
            focus_group_index=[True, False],
            focus_group_name="Brand A",
            comparison_group_index=[False, True],
            comparison_group_name="Brand B",
        )


def test_invalid_agg_func_raises(sample_df):
    """Test that GainLoss raises ValueError for invalid aggregation function."""
    with pytest.raises(ValueError, match="Aggregation function .* not supported"):
        GainLoss(
            df=sample_df,
            p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
            p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
            focus_group_index=sample_df["brand"] == "Brand A",
            focus_group_name="Brand A",
            comparison_group_index=sample_df["brand"] == "Brand B",
            comparison_group_name="Brand B",
            agg_func="invalid_function",  # Invalid aggregation
        )


def test_lazy_evaluation_of_df_property(sample_df):
    """Test that .df property provides lazy evaluation and caching."""
    ibis_table = ibis.memtable(sample_df)
    gl = GainLoss(
        df=ibis_table,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )

    assert isinstance(gl.table, ibis.Table)
    assert gl._df is None

    # First access triggers execution
    result1 = gl.df
    assert isinstance(result1, pd.DataFrame)
    assert gl._df is not None

    # Second access returns cached result
    result2 = gl.df
    assert result2 is result1  # Same object reference


def test_lazy_evaluation_of_backward_compatibility_properties(sample_df):
    """Test that backward compatibility properties also provide lazy evaluation and caching."""
    ibis_table = ibis.memtable(sample_df)
    gl = GainLoss(
        df=ibis_table,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )

    # Verify backward compatibility properties are not pre-computed
    assert not hasattr(gl, "_gain_loss_df")

    # First access to gain_loss_df triggers execution and caching
    result1 = gl.gain_loss_df
    assert isinstance(result1, pd.DataFrame)
    assert hasattr(gl, "_gain_loss_df")

    # Second access returns cached result
    result2 = gl.gain_loss_df
    assert result2 is result1  # Same object reference

    # First access to df triggers execution and caching
    result3 = gl.df
    assert isinstance(result3, pd.DataFrame)
    assert gl._df is not None

    # Second access returns cached result
    result4 = gl.df
    assert result4 is result3  # Same object reference


def test_backward_compatibility_attributes(sample_df):
    """Test that backward compatibility attributes exist and work correctly."""
    gl = GainLoss(
        df=sample_df,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )

    # Old attributes should exist and be accessible as properties
    assert hasattr(gl, "gain_loss_df")

    # Should be pandas DataFrames
    assert isinstance(gl.gain_loss_df, pd.DataFrame)
    assert isinstance(gl.df, pd.DataFrame)

    # gain_loss_df should be customer-level data (more rows than aggregated)
    assert len(gl.gain_loss_df) >= len(gl.df)

    # Both should have non-empty content
    assert not gl.gain_loss_df.empty
    assert not gl.df.empty


def test_gainloss_with_group_col_end_to_end(sample_df):
    """Test GainLoss with group_col parameter for grouped analysis."""
    # Add a category column to sample data
    sample_df = sample_df.copy()
    sample_df["category"] = ["Cat1", "Cat2"] * 6

    gl = GainLoss(
        df=sample_df,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
        group_col="category",
    )

    result = gl.df
    assert "category" in result.columns
    category_len = 2
    assert len(result) == category_len  # Two categories

    # Verify category values are present
    categories = result["category"].unique()
    assert set(categories) == {"Cat1", "Cat2"}

    # Verify all expected columns exist in grouped result
    expected_cols = [
        "category",
        "focus_p1",
        "comparison_p1",
        "total_p1",
        "focus_p2",
        "comparison_p2",
        "total_p2",
        "focus_diff",
        "comparison_diff",
        "total_diff",
        "new",
        "lost",
        "increased_focus",
        "decreased_focus",
        "switch_from_comparison",
        "switch_to_comparison",
    ]
    for col in expected_cols:
        assert col in result.columns

    # Verify backward compatibility attributes also work with grouping
    assert isinstance(gl.gain_loss_df, pd.DataFrame)
    assert isinstance(gl.df, pd.DataFrame)
    assert "category" in gl.gain_loss_df.columns


def test_apply_business_logic_ibis():
    """Test _apply_business_logic_ibis method directly (core Ibis conversion from ticket #152)."""
    # Create test data that covers all business logic scenarios
    expected_new_value = 50
    expected_lost_value = -100

    test_data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "focus_p1": [0, 100, 100, 100, 100],  # New, existing, existing, existing, existing
            "comparison_p1": [0, 0, 100, 100, 100],  # New, existing, existing, existing, existing
            "focus_p2": [expected_new_value, 0, 120, 80, 110],  # New, lost, increased, decreased, switched
            "comparison_p2": [0, 0, 110, 90, 150],  # New, lost, increased, decreased, switched
            "focus_diff": [expected_new_value, expected_lost_value, 20, -20, 10],
            "comparison_diff": [0, 0, 10, -10, 50],
        },
    )

    ibis_table = ibis.memtable(test_data)
    result_table = GainLoss._apply_business_logic_ibis(ibis_table)
    result_df = result_table.execute()

    # Verify new columns exist
    expected_new_cols = [
        "new",
        "lost",
        "increased_focus",
        "decreased_focus",
        "switch_from_comparison",
        "switch_to_comparison",
    ]
    for col in expected_new_cols:
        assert col in result_df.columns

    # Customer 1: New customer (P1 = 0,0 -> P2 = 50,0)
    assert result_df.iloc[0]["new"] == expected_new_value
    assert result_df.iloc[0]["lost"] == 0

    # Customer 2: Lost customer (P1 = 100,0 -> P2 = 0,0)
    assert result_df.iloc[1]["new"] == 0
    assert result_df.iloc[1]["lost"] == expected_lost_value

    # Customer 3: Existing customer with increases
    assert result_df.iloc[2]["new"] == 0
    assert result_df.iloc[2]["lost"] == 0
    assert result_df.iloc[2]["increased_focus"] >= 0


def test_ibis_vs_pandas_result_equivalence(sample_df):
    """Test that Ibis and pandas implementations produce identical results (ticket requirement)."""
    p1_index = sample_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2_index = sample_df["transaction_date"] >= datetime.date(2023, 5, 1)
    focus_group_index = sample_df["brand"] == "Brand A"
    comparison_group_index = sample_df["brand"] == "Brand B"

    # Create GainLoss with pandas DataFrame
    gl_pandas = GainLoss(
        df=sample_df,
        p1_index=p1_index,
        p2_index=p2_index,
        focus_group_index=focus_group_index,
        focus_group_name="Brand A",
        comparison_group_index=comparison_group_index,
        comparison_group_name="Brand B",
    )

    # Create GainLoss with Ibis table
    ibis_table = ibis.memtable(sample_df)
    gl_ibis = GainLoss(
        df=ibis_table,
        p1_index=p1_index,
        p2_index=p2_index,
        focus_group_index=focus_group_index,
        focus_group_name="Brand A",
        comparison_group_index=comparison_group_index,
        comparison_group_name="Brand B",
    )

    # Results should be identical
    pandas_result = gl_pandas.df
    ibis_result = gl_ibis.df

    # Compare columns
    assert set(pandas_result.columns) == set(ibis_result.columns)

    # Compare data (sort both for consistent comparison)
    pandas_sorted = pandas_result.sort_values(list(pandas_result.columns)).reset_index(drop=True)
    ibis_sorted = ibis_result.sort_values(list(ibis_result.columns)).reset_index(drop=True)

    # Use approximate equality for float columns
    pd.testing.assert_frame_equal(pandas_sorted, ibis_sorted, check_dtype=False)
