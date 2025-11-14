"""Unified integration tests for HML Segmentation with multiple database backends."""

import pytest

from pyretailscience.segmentation.hml import HMLSegmentation


@pytest.mark.parametrize(
    "zero_value_customers",
    ["separate_segment", "include_with_light", "exclude"],
)
def test_hml_segmentation_integration(
    transactions_table,
    zero_value_customers,
):
    """Integration test for HMLSegmentation using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        zero_value_customers: Parameter for handling zero-value customers
    """
    limited_table = transactions_table.limit(5000)

    hml_segmentation = HMLSegmentation(
        df=limited_table,
        value_col="unit_cost",
        agg_func="mean",
        zero_value_customers=zero_value_customers,
    )

    assert hml_segmentation is not None
    result = hml_segmentation.df
    assert result is not None
