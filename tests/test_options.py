"""Tests for the Options module."""

from pathlib import Path
from unittest.mock import patch

import pytest
import toml

import pyretailscience.options as opt


class TestOptions:
    """Test for option handling class."""

    def test_unknown_option_raises_value_error(self):
        """Test setting/getting/resetting an unknown option raises a ValueError."""
        options = opt.Options()
        with pytest.raises(ValueError, match="Unknown option: unknown.option"):
            options.set_option("unknown.option", "some_value")
        with pytest.raises(ValueError, match="Unknown option: unknown.option"):
            options.get_option("unknown_option")
        with pytest.raises(ValueError, match="Unknown option: unknown.option"):
            options.reset_option("unknown_option")
        with pytest.raises(ValueError, match="Unknown option: unknown.option"):
            options.describe_option("unknown_option")

    def test_list_options_returns_all_options(self):
        """Test listing all options returns all options."""
        options = opt.Options()
        assert options.list_options() == list(options._options.keys())

    def test_set_option_updates_value(self):
        """Test setting an option updates the option value correctly."""
        options = opt.Options()
        options.set_option("column.customer_id", "new_customer_id")
        assert options.get_option("column.customer_id") == "new_customer_id"

    def test_get_option_retrieves_correct_value(self):
        """Test getting an option retrieves the correct value."""
        options = opt.Options()
        expected_value = options._options["column.customer_id"]
        actual_value = options.get_option("column.customer_id")
        assert actual_value == expected_value

    def test_reset_option_restores_default_value(self):
        """Test resetting an option restores its default value."""
        options = opt.Options()
        expected_value = options._options["column.customer_id"]
        options.set_option("column.customer_id", "new_customer_id")
        options.reset_option("column.customer_id")
        assert options.get_option("column.customer_id") == expected_value

    def test_describe_option_correct_description_and_value(self):
        """Test describing an option provides the correct description and current value."""
        options = opt.Options()
        option = "column.customer_id"
        expected_description = options._descriptions[option]
        expected_value = options._options[option]

        description = options.describe_option(option)
        assert description == f"{option}: {expected_description} (current value: {expected_value})"

    def test_matching_keys_between_options_and_descriptions(self):
        """Test that all options have a corresponding description and vice versa."""
        options = opt.Options()
        assert set(options._options.keys()) == set(options._descriptions.keys())

    def test_context_manager_overrides_option(self):
        """Test that the context manager overrides the option value correctly at the global level."""
        original_value = opt.get_option("column.customer_id")
        with opt.option_context("column.customer_id", "new_customer_id"):
            assert opt.get_option("column.customer_id") == "new_customer_id"
        assert opt.get_option("column.customer_id") == original_value

    def test_context_manager_odd_number_of_arguments_raises_value_error(self):
        """Test that the context manager raises a ValueError when an odd number of arguments is passed."""
        with (
            pytest.raises(ValueError, match="The context manager requires an even number of arguments"),
            opt.option_context("column.customer_id"),
        ):
            pass

    def test_set_option_updates_value_global_level(self):
        """Test setting an option updates the option value correctly at the global level."""
        opt.set_option("column.customer_id", "new_customer_id")
        assert opt.get_option("column.customer_id") == "new_customer_id"
        opt.reset_option("column.customer_id")

    def test_get_option_retrieves_correct_value_global_level(self):
        """Test getting an option retrieves the correct value at the global level."""
        # Instantiate Options class to get the default value
        options = opt.Options()
        expected_value = options._options["column.customer_id"]
        del options

        actual_value = opt.get_option("column.customer_id")
        assert actual_value == expected_value

    def test_reset_option_restores_default_value_global_level(self):
        """Test resetting an option restores its default value at the global level."""
        # Instantiate Options class to get the default value
        options = opt.Options()
        expected_value = options._options["column.customer_id"]
        del options

        opt.set_option("column.customer_id", "new_customer_id")
        opt.reset_option("column.customer_id")
        assert opt.get_option("column.customer_id") == expected_value

    def test_describe_option_correct_description_and_value_global_level(self):
        """Test describing an option provides the correct description and current value at the global level."""
        option = "column.customer_id"
        # Instantiate Options class to get the default value
        options = opt.Options()
        expected_description = options._descriptions[option]
        expected_value = options._options[option]
        del options

        description = opt.describe_option(option)
        assert description == f"{option}: {expected_description} (current value: {expected_value})"

    def test_list_options_returns_all_options_global_level(self):
        """Test listing all options returns all options at the global level."""
        options = opt.Options()
        options_list = list(options._options.keys())
        del options

        assert opt.list_options() == options_list

    def test_load_invalid_format_toml(self):
        """Test loading an invalid TOML file raises a ValueError."""
        test_file_path = Path("tests/toml_files/corrupt.toml").resolve()
        with pytest.raises(toml.TomlDecodeError):
            opt.Options.load_from_toml(test_file_path)

    def test_load_valid_toml(self):
        """Test loading a valid TOML file updates the options correctly."""
        test_file_path = Path("tests/toml_files/valid.toml").resolve()
        options = opt.Options.load_from_toml(test_file_path)
        assert options.get_option("column.customer_id") == "new_customer_id"
        assert options.get_option("column.product_id") == "new_product_id"
        assert options.get_option("column.agg.customer_id") == "new_customers"
        assert options.get_option("column.calc.price_per_unit") == "new_price_per_unit"
        assert options.get_option("column.suffix.count") == "new_cnt"

    def test_load_invalid_option_toml(self):
        """Test loading an invalid TOML file raises a ValueError."""
        test_file_path = Path("tests/toml_files/invalid_option.toml").resolve()
        with pytest.raises(ValueError, match="Unknown option in TOML file: column.agg.unknown_column"):
            opt.Options.load_from_toml(test_file_path)

    def test_flatten_options(self):
        """Test flattening the options dictionary."""
        nested_options = {
            "column": {
                "customer_id": "customer_id",
                "agg": {
                    "customer_id": "customer_id",
                    "product_id": "product_id",
                },
            },
        }
        expected_flat_options = {
            "column.customer_id": "customer_id",
            "column.agg.customer_id": "customer_id",
            "column.agg.product_id": "product_id",
        }
        assert expected_flat_options == opt.Options.flatten_options("column", nested_options["column"])

    @pytest.fixture
    def _reset_lru_cache(self):
        opt.find_project_root.cache_clear()
        yield
        opt.find_project_root.cache_clear()

    @pytest.mark.usefixtures("_reset_lru_cache")
    @patch("pathlib.Path.cwd")
    @patch("pathlib.Path.is_dir")
    def test_find_project_root_git_found(self, mock_is_dir, mock_cwd):
        """Test finding the project root when the .git directory is found."""
        mock_cwd.return_value = Path("/home/user/project")
        mock_is_dir.side_effect = [True]  # .git directory exists
        assert opt.find_project_root() == Path("/home/user/project")

    @pytest.mark.usefixtures("_reset_lru_cache")
    @patch("pathlib.Path.cwd")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.is_file")
    def test_find_project_root_toml_found(self, mock_is_file, mock_is_dir, mock_cwd):
        """Test finding the project root when the pyretailscience.toml file is found."""
        mock_cwd.return_value = Path("/home/user/project")
        mock_is_dir.side_effect = [False]  # .git directory doesn't exist
        mock_is_file.side_effect = [True]  # pyretailscience.toml file exists
        assert opt.find_project_root() == Path("/home/user/project")

    @pytest.mark.usefixtures("_reset_lru_cache")
    @patch("pathlib.Path.cwd")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.is_file")
    @patch("pathlib.Path.parent")
    def test_find_project_root_no_project_found(self, mock_parent, mock_is_file, mock_is_dir, mock_cwd):
        """Test finding the project root when no project root is found."""
        mock_cwd.return_value = Path("/")
        mock_is_dir.side_effect = [False, False]  # No .git directory
        mock_is_file.side_effect = [False, False]  # No pyretailscience.toml file
        mock_parent.return_value = Path("/")
        assert opt.find_project_root() is None

    @pytest.mark.usefixtures("_reset_lru_cache")
    @patch("pathlib.Path.cwd")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.is_file")
    def test_find_project_root_found_in_parent(self, mock_is_file, mock_is_dir, mock_cwd):
        """Test finding the project root when the project root is found in a parent directory."""
        mock_cwd.return_value = Path("/home/user/project/subdir")
        mock_is_dir.side_effect = [False, True]  # .git directory in parent
        mock_is_file.side_effect = [False]  # No pyretailscience.toml file
        assert opt.find_project_root() == Path("/home/user/project")

    def test_load_option_toml(self):
        """Test loading the test_options.toml file updates the options correctly."""
        test_file_path = Path("tests/toml_files/test_options.toml").resolve()
        options = opt.Options.load_from_toml(test_file_path)

        assert options.get_option("column.customer_id") == "new_customer_id"
        assert options.get_option("column.transaction_id") == "new_transaction_id"
        assert options.get_option("column.transaction_date") == "new_transaction_date"
        assert options.get_option("column.transaction_time") == "new_transaction_time"
        assert options.get_option("column.product_id") == "new_product_id"
        assert options.get_option("column.unit_quantity") == "new_unit_quantity"
        assert options.get_option("column.unit_price") == "new_unit_price"
        assert options.get_option("column.unit_spend") == "new_unit_spend"
        assert options.get_option("column.unit_cost") == "new_unit_cost"
        assert options.get_option("column.promo_unit_spend") == "new_promo_unit_spend"
        assert options.get_option("column.promo_unit_quantity") == "new_promo_unit_quantity"
        assert options.get_option("column.store_id") == "new_store_id"

        assert options.get_option("column.agg.customer_id") == "new_customers"
        assert options.get_option("column.agg.transaction_id") == "new_transactions"
        assert options.get_option("column.agg.product_id") == "new_products"
        assert options.get_option("column.agg.unit_quantity") == "new_units"
        assert options.get_option("column.agg.unit_price") == "new_prices"
        assert options.get_option("column.agg.unit_spend") == "new_spend"
        assert options.get_option("column.agg.unit_cost") == "new_costs"
        assert options.get_option("column.agg.promo_unit_spend") == "new_promo_spend"
        assert options.get_option("column.agg.promo_unit_quantity") == "new_promo_units"
        assert options.get_option("column.agg.store_id") == "new_stores"

        assert options.get_option("column.calc.price_per_unit") == "new_price_per_unit"
        assert options.get_option("column.calc.units_per_transaction") == "new_units_per_transaction"
        assert options.get_option("column.calc.spend_per_customer") == "new_spend_per_customer"
        assert options.get_option("column.calc.spend_per_transaction") == "new_spend_per_transaction"
        assert options.get_option("column.calc.transactions_per_customer") == "new_transactions_per_customer"
        assert options.get_option("column.calc.price_elasticity") == "new_price_elasticity"

        assert options.get_option("column.suffix.count") == "new_cnt"
        assert options.get_option("column.suffix.percent") == "new_pct"
        assert options.get_option("column.suffix.difference") == "new_diff"
        assert options.get_option("column.suffix.percent_difference") == "new_pct_diff"
        assert options.get_option("column.suffix.contribution") == "new_contrib"
        assert options.get_option("column.suffix.period_1") == "new_p1"
        assert options.get_option("column.suffix.period_2") == "new_p2"
