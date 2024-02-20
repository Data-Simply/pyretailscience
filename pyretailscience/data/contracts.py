import abc
import re
from enum import Enum

import numpy as np
import pandas as pd
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import PandasDataset


class EValidationState(Enum):
    """
    Enumeration representing the validation state of a contract.

    Attributes:
        VALID (str): Represents a valid contract.
        INVALID (str): Represents an invalid contract.
        UNKNOWN (str): Represents an unknown validation state.
    """

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class EExpectationSet(Enum):
    """
    Enumeration representing the expectation sets.

    **Basic Expectations** are a set of expectations that can be quickly run, such as checking whether a column exists in a DataFrame.
    **Extended Expectations** are a set of expectations that are more computationally expensive, such as checking whether a column has unique values.

    Attributes:
        BASIC (str): Basic expectation set.
        EXTENDED (str): Extended expectation set.
    """

    BASIC = "basic"
    EXTENDED = "extended"


class PyRetailSciencePandasDataset(PandasDataset):
    """A subclass of PandasDataset that adds custom expectations for the PyRetailScience project based on its data
    specs.

    Attributes:
        _data_asset_type (str): The data asset type for the PyRetailSciencePandasDataset.
    """

    _data_asset_type = "PyRetailSciencePandasDataset"

    @PandasDataset.expectation(["self"])
    def expect_product_and_quantity_sign_to_be_unique_in_a_transaction(
        self,
    ) -> dict[str, any]:
        """Asserts that the combination of transaction_id, product_id, and quantity sign are unique. This is useful to
        ensure that the same product is not added or removed more than once in the same transaction. The data contract
        specifies that the quantity sign should be unique for each transaction_id and product_id combination. Returns
        should have negative quanity sign and purchases should have positive quantity sign.

        Returns:
            dict[str, any]: A dictionary with the success of the expectation, and the number of elements, missing
                elements, and missing percentage in the DataFrame. For example:
                {
                    "success": True,
                    "result": {
                        "element_count": 100,
                        "missing_count": 0,
                        "missing_percent": 0.0
                    }
                }
        """
        grouped_df = self.groupby(["transaction_id", "product_id", np.sign(self["quantity"])]).size()
        nulls = self[["transaction_id", "product_id", "quantity"]].isnull()
        return {
            "success": all(grouped_df == 1),
            "result": {
                "element_count": len(self),
                "missing_count": nulls.sum().sum(),
                "missing_percent": nulls.mean().mean(),
            },
        }


class ContractBase(abc.ABC):
    """Base class for data contracts. It contains the basic and extended expectations for the data, as well as the
    validation state and the result of the last validation. It also contains a method to validate the data.

    Attributes:
        basic_expectations (list[ExpectationConfiguration]): A list of basic expectations for the data.
        extended_expectations (list[ExpectationConfiguration]): A list of extended expectations for the data.
        validation_state (EValidationState): The validation state of the data.
        expectations_run (EExpectationSet | None): The expectation set that was run in the last validation.
        validation_result (dict): The result of the last validation.

    Args:
        df (pd.DataFrame): The DataFrame to be validated.
    """

    basic_expectations: list[ExpectationConfiguration] = []
    extended_expectations: list[ExpectationConfiguration] = []
    validation_state: EValidationState = EValidationState.UNKNOWN
    expectations_run: EExpectationSet | None = None
    validation_result: dict = {}

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def validate(
        self,
        expectation_set: EExpectationSet = EExpectationSet.BASIC,
        verbose: bool = False,
    ) -> bool:
        """Validates the data against the expectations.

        Args:
            expectation_set (EExpectationSet, optional): The expectation set to run. Defaults to EExpectationSet.BASIC.
            verbose (bool, optional): Whether to print the failed expectations. Defaults to False.

        Returns:
            bool: True if the data meets the expectations in the selected expectation_set, False otherwise.
        """

        # Ensure it's an enum
        expectation_set = EExpectationSet(expectation_set)

        # Return true if the data is already valid
        if self.validation_state == EValidationState.VALID:
            if self.expectations_run == EExpectationSet.EXTENDED or (
                expectation_set == EExpectationSet.BASIC and self.expectations_run == EExpectationSet.BASIC
            ):
                return True

        expectations = self.basic_expectations.copy()
        if expectation_set == EExpectationSet.EXTENDED:
            expectations += self.extended_expectations

        gx_df = PyRetailSciencePandasDataset(self._df)
        results = gx_df.validate(
            expectation_suite=ExpectationSuite(expectation_suite_name="expectations", expectations=expectations)
        )
        self.validation_result = results

        self.validation_state = EValidationState.INVALID
        if results["success"]:
            self.validation_state = EValidationState.VALID
            self.expectations_run = expectation_set
        else:
            if verbose:
                for i in results["results"]:
                    if not i["success"]:
                        print(i["expectation_config"]["expectation_type"])
                        print(i["expectation_config"]["kwargs"])
                        exception_message = i["exception_info"]["exception_message"]
                        if exception_message:
                            print(exception_message)
                        exception_traceback = i["exception_info"]["exception_traceback"]
                        if exception_traceback:
                            print(exception_traceback)

        return results["success"]


class TransactionLevelContract(ContractBase):
    """Data contract for the transaction level data. It contains the basic and extended expectations for the data, as
    well as the validation state and the result of the last validation. It also contains a method to validate the data.

    Attributes:
        basic_expectations (list[ExpectationConfiguration]): A list of basic expectations for the data. These include
            expectations that the columns transaction_id, transaction_datetime, customer_id, total_price, and store_id
            exist.
        extended_expectations (list[ExpectationConfiguration]): A list of extended expectations for the data. These
            include expectations that the transaction_id, transaction_datetime, customer_id, total_price, and store_id
            columns have unique values, that the transaction_datetime column has values between 1970-01-01 and
            2029-12-31, and that the transaction_id, transaction_datetime, customer_id, and store_id columns have unique
            combinations.
        validation_state (EValidationState): The validation state of the data.
        expectations_run (EExpectationSet | None): The expectation set that was run in the last validation.
        validation_result (dict): The result of the last validation.
    """

    basic_expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "transaction_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "transaction_datetime"},
        ),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "customer_id"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "total_price"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "store_id"}),
    ]

    extended_expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "transaction_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_compound_columns_to_be_unique",
            kwargs={
                "column_list": [
                    "transaction_id",
                    "transaction_datetime",
                    "customer_id",
                    "store_id",
                ]
            },
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "transaction_datetime",
                "min_value": "1970-01-01",
                "max_value": "2029-12-31",
            },
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "transaction_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "transaction_datetime"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "customer_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "total_price"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "store_id"},
        ),
    ]


class TransactionItemLevelContract(ContractBase):
    """Data contract for the transaction item level data. It contains the basic and extended expectations for the data,
    as well as the validation state and the result of the last validation. It also contains a method to validate the
    data.

    Attributes:
        basic_expectations (list[ExpectationConfiguration]): A list of basic expectations for the data. These include
            expectations that the columns transaction_id, transaction_datetime, customer_id, total_price, store_id,
            product_id, product_name, unit_price, and quantity exist.
        extended_expectations (list[ExpectationConfiguration]): A list of extended expectations for the data. These
            include expectations that the transaction_id, transaction_datetime, customer_id, store_id, and product_id
            columns have unique values, that the transaction_id, transaction_datetime, customer_id, and store_id columns
            have unique combinations, and that the product_id, product_name, and quantity columns have no null values.
        validation_state (EValidationState): The validation state of the data.
        expectations_run (EExpectationSet | None): The expectation set that was run in the last validation.
        validation_result (dict): The result of the last validation.

    Args:
        df (pd.DataFrame): The DataFrame to be validated. If category or brand columns are present, it adds expectations
            that these columns are not null.
    """

    basic_expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "transaction_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "transaction_datetime"},
        ),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "customer_id"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "total_price"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "store_id"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "product_id"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "product_name"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "unit_price"}),
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "quantity"}),
    ]

    extended_expectations = [
        ExpectationConfiguration(
            expectation_type="expect_compound_columns_to_be_unique",
            kwargs={
                "column_list": [
                    "transaction_id",
                    "transaction_datetime",
                    "customer_id",
                    "store_id",
                ]
            },
        ),
        ExpectationConfiguration(
            expectation_type="expect_transaction_product_quantity_sign_to_be_unique",
            kwargs={},
        ),
        # Null expectations
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "transaction_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "transaction_datetime"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "customer_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "total_price"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "store_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "product_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "product_name"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "unit_price"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "quantity"},
        ),
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        # If category or brand columns are present, add expectations for them
        category_pattern = re.compile(r"category_\d+_(id|name)")
        category_matches = [s for s in df.columns if category_pattern.match(s)]
        for match in category_matches:
            self.extended_expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": match},
                )
            )

        if "brand_id" or "brand_name" in df.columns:
            self.extended_expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "brand_id"},
                )
            )
            self.extended_expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "brand_name"},
                )
            )
        super().__init__(df)


class CustomerLevelContract(ContractBase):
    """Data contract for the customer level data. It contains the basic and extended expectations for the data, as well
    as the validation state and the result of the last validation. It also contains a method to validate the data.

    Attributes:
        basic_expectations (list[ExpectationConfiguration]): A list of basic expectations for the data. These include
            expectations that the column customer_id exists.
        extended_expectations (list[ExpectationConfiguration]): A list of extended expectations for the data. These
            include expectations that the customer_id column has unique values, and that the customer_id column has no
            null values.
        validation_state (EValidationState): The validation state of the data.
        expectations_run (EExpectationSet | None): The expectation set that was run in the last validation.
        validation_result (dict): The result of the last validation.
    """

    basic_expectations = [
        ExpectationConfiguration(expectation_type="expect_column_to_exist", kwargs={"column": "customer_id"}),
    ]

    extended_expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "customer_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "customer_id"},
        ),
    ]
