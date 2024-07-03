"""Tests for the data contracts module."""

import numpy as np
import pandas as pd
import pytest
from great_expectations.core.expectation_configuration import ExpectationConfiguration

from pyretailscience.data import contracts


@pytest.fixture()
def dataset():
    """Create a sample dataset for testing."""
    data = {
        "transaction_id": [1, 1, 2, 2, 3],
        "product_id": ["A", "B", "A", "B", "C"],
        "quantity": [1, -1, 2, -2, 3],
    }
    return pd.DataFrame(data)


def test_expect_product_and_quantity_sign_to_be_unique_in_a_transaction(dataset):
    """Test the expect_product_and_quantity_sign_to_be_unique_in_a_transaction function."""
    test_dataset = contracts.PyRetailSciencePandasDataset(dataset)

    element_count = len(test_dataset)
    result = test_dataset.expect_product_and_quantity_sign_to_be_unique_in_a_transaction()
    assert result["success"] is True
    assert result["result"]["element_count"] == element_count
    assert result["result"]["missing_count"] == 0
    assert result["result"]["missing_percent"] == 0.0

    # Test case where the combination of transaction_id, product_id, and quantity sign is not unique
    test_dataset.loc[5] = [1, "A", 1]
    element_count = len(test_dataset)
    result = test_dataset.expect_product_and_quantity_sign_to_be_unique_in_a_transaction()
    assert result["success"] is False
    assert result["result"]["element_count"] == element_count
    assert result["result"]["missing_count"] == 0
    assert result["result"]["missing_percent"] == 0.0

    # Test case where there are missing values
    test_dataset.loc[6] = [2, "B", np.nan]
    element_count = len(test_dataset)
    missing_percent = test_dataset.isna().sum().sum() / test_dataset.size

    result = test_dataset.expect_product_and_quantity_sign_to_be_unique_in_a_transaction()
    assert result["success"] is False
    assert result["result"]["element_count"] == element_count
    assert result["result"]["missing_count"] == 1
    assert result["result"]["missing_percent"] == missing_percent


def test_validate_contract_base(dataset):
    """Test the validate function of the ContractBase class."""
    test_contract = contracts.ContractBase(dataset)

    assert test_contract.validation_state == contracts.EValidationState.UNKNOWN
    assert test_contract.expectations_run is None
    assert test_contract.validation_result == {}

    # Test case where basic expectations are validated successfully
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)
    assert result is True
    assert test_contract.validation_state == contracts.EValidationState.VALID
    assert test_contract.expectations_run == contracts.EExpectationSet.BASIC

    # Test case where extended expectations are validated successfully
    test_contract = contracts.ContractBase(dataset)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.EXTENDED)
    assert result is True
    assert test_contract.validation_state == contracts.EValidationState.VALID
    assert test_contract.expectations_run == contracts.EExpectationSet.EXTENDED


def test_build_expected_columns():
    """Test the build_expected_columns function."""
    columns = ["column1", "column2", "column3"]
    expected_expectations = [
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "column1"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "column2"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "column3"}),
    ]

    expectations = contracts.build_expected_columns(columns)

    assert expectations == expected_expectations

    # Test with a positive case dataframe where all the columns exists
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6], "column3": [7, 8, 9]})

    class TestContract(contracts.ContractBase):
        basic_expectations = expectations

    test_contract = TestContract(df)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)

    assert result is True

    # Test with a negative case dataframe where one of the columns is missing
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
    test_contract = TestContract(df)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)

    assert result is False


def test_build_expected_unique_columns():
    """Test the build_expected_unique_columns function."""
    columns = ["column1", "column2"]
    expected_expectations = [
        ExpectationConfiguration(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "column1"}),
        ExpectationConfiguration(expectation_type="expect_column_values_to_be_unique", kwargs={"column": "column2"}),
    ]

    expectations = contracts.build_expected_unique_columns(columns)

    assert expectations == expected_expectations

    # Test with a positive case dataframe where all the columns are unique
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [7, 8, 9]})

    class TestContract(contracts.ContractBase):
        basic_expectations = expectations

    test_contract = TestContract(df)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)

    assert result is True

    # Test with a negative case dataframe where one of the columns is not unique
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [7, 8, 7]})

    test_contract = TestContract(df)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)

    assert result is False


def test_build_non_null_columns():
    """Test the build_non_null_columns function."""
    columns = ["column1", "column2"]
    expected_expectations = [
        ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "column1"}),
        ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null", kwargs={"column": "column2"}),
    ]

    expectations = contracts.build_non_null_columns(columns)

    assert expectations == expected_expectations

    # Test with a positive case dataframe where all the columns have no null values
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})

    class TestContract(contracts.ContractBase):
        basic_expectations = expectations

    test_contract = TestContract(df)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)

    assert result is True

    # Test with a negative case dataframe where one of the columns has null values
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, None, 6]})

    test_contract = TestContract(df)
    result = test_contract.validate(expectation_set=contracts.EExpectationSet.BASIC)

    assert result is False
