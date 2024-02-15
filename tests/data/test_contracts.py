import numpy as np
import pandas as pd
import pytest

from pyretailscience.data import contracts


@pytest.fixture
def dataset():
    # Create a sample dataset for testing
    data = {"transaction_id": [1, 1, 2, 2, 3], "product_id": ["A", "B", "A", "B", "C"], "quantity": [1, -1, 2, -2, 3]}
    return pd.DataFrame(data)


def test_expect_product_and_quantity_sign_to_be_unique_in_a_transaction(dataset):
    # Test case where the combination of transaction_id, product_id, and quantity sign is unique
    test_dataset = contracts.PyRetailSciencePandasDataset(dataset)

    result = test_dataset.expect_product_and_quantity_sign_to_be_unique_in_a_transaction()
    assert result["success"] is True
    assert result["result"]["element_count"] == 5
    assert result["result"]["missing_count"] == 0
    assert result["result"]["missing_percent"] == 0.0

    # Test case where the combination of transaction_id, product_id, and quantity sign is not unique
    test_dataset.loc[5] = [1, "A", 1]
    result = test_dataset.expect_product_and_quantity_sign_to_be_unique_in_a_transaction()
    assert result["success"] is False
    assert result["result"]["element_count"] == 6
    assert result["result"]["missing_count"] == 0
    assert result["result"]["missing_percent"] == 0.0

    # Test case where there are missing values
    test_dataset.loc[6] = [2, "B", np.nan]
    result = test_dataset.expect_product_and_quantity_sign_to_be_unique_in_a_transaction()
    assert result["success"] is False
    assert result["result"]["element_count"] == 7
    assert result["result"]["missing_count"] == 1
    assert round(result["result"]["missing_percent"], 4) == 0.0476


def test_validate_contract_base(dataset):

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
