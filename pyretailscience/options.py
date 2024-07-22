"""This module provides a simplified implementation of a pandas-like options system.

It allows users to get, set, and reset various options that control the behavior
of data display and processing. The module also includes a context manager for
temporarily changing options.

Example:
    >>> set_option('display.max_rows', 100)
    >>> print(get_option('display.max_rows'))
    100
    >>> with option_context('display.max_rows', 10):
    ...     print(get_option('display.max_rows'))
    10
    >>> print(get_option('display.max_rows'))
    100

"""

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import toml

OptionTypes = str | int | float | bool | list | dict | None


class Options:
    """A class to manage configurable options."""

    def __init__(self) -> None:
        """Initializes the options with default values."""
        self._options: dict[str, OptionTypes] = {
            # Database columns
            "column.customer_id": "customer_id",
            "column.transaction_id": "transaction_id",
            "column.transaction_date": "transaction_date",
            "column.transaction_time": "transaction_time",
            "column.product_id": "product_id",
            "column.unit_quantity": "unit_quantity",
            "column.unit_price": "unit_price",
            "column.unit_spend": "unit_spend",
            "column.store_id": "store_id",
            # Aggregation columns
            "column.agg.customer_id": "customers",
            "column.agg.transaction_id": "transactions",
            "column.agg.product_id": "products",
            "column.agg.unit_quantity": "units",
            "column.agg.unit_price": "prices",
            "column.agg.unit_spend": "spend",
            "column.agg.store_id": "stores",
            # Calculated columns
            "column.calc.price_per_unit": "price_per_unit",
            "column.calc.units_per_transaction": "units_per_transaction",
            # Abbreviation suffix
            "column.suffix.count": "cnt",
            "column.suffix.percent": "pct",
            "column.suffix.difference": "diff",
            "column.suffix.contribution": "contrib",
        }
        self._descriptions: dict[str, str] = {
            # Database columns
            "column.customer_id": "The name of the column containing customer IDs.",
            "column.transaction_id": "The name of the column containing transaction IDs.",
            "column.transaction_date": "The name of the column containing transaction dates.",
            "column.transaction_time": "The name of the column containing transaction times.",
            "column.product_id": "The name of the column containing product IDs.",
            "column.unit_quantity": "The name of the column containing the number of units sold.",
            "column.unit_price": "The name of the column containing the unit price of the product.",
            "column.unit_spend": (
                "The name of the column containing the total spend of the products in the transaction."
                "ie, unit_price * units",
            ),
            "column.store_id": "The name of the column containing store IDs of the transaction.",
            # Aggregation columns
            "column.agg.customer_id": "The name of the column containing the number of unique customers.",
            "column.agg.transaction_id": "The name of the column containing the number of transactions.",
            "column.agg.product_id": "The name of the column containing the number of unique products.",
            "column.agg.unit_quantity": "The name of the column containing the total number of units sold.",
            "column.agg.unit_price": "The name of the column containing the average unit price of products.",
            "column.agg.unit_spend": (
                "The name of the column containing the total spend of the units in the transaction."
            ),
            "column.agg.store_id": "The name of the column containing the number of unique stores.",
            # Calculated columns
            "column.calc.price_per_unit": "The name of the column containing the price per unit.",
            "column.calc.units_per_transaction": "The name of the column containing the units per transaction.",
            # Abbreviation suffixes
            "column.suffix.count": "The suffix to use for count columns.",
            "column.suffix.percent": "The suffix to use for percentage columns.",
            "column.suffix.difference": "The suffix to use for difference columns.",
            "column.suffix.contribution": "The suffix to use for revenue contribution columns.",
        }
        self._default_options: dict[str, OptionTypes] = self._options.copy()

    def set_option(self, pat: str, val: OptionTypes) -> None:
        """Set the value of the specified option.

        Args:
            pat: The option name.
            val: The value to set the option to.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat not in self._options:
            msg = f"Unknown option: {pat}"
            raise ValueError(msg)

        self._options[pat] = val

    def get_option(self, pat: str) -> OptionTypes:
        """Get the value of the specified option.

        Args:
            pat: The option name.

        Returns:
            The value of the option.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat in self._options:
            return self._options[pat]

        msg = f"Unknown option: {pat}"
        raise ValueError(msg)

    def reset_option(self, pat: str) -> None:
        """Reset the specified option to its default value.

        Args:
            pat: The option name.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat not in self._options:
            msg = f"Unknown option: {pat}"
            raise ValueError(msg)

        self._options[pat] = self._default_options[pat]

    def list_options(self) -> list[str]:
        """List all available options.

        Returns:
            A list of all option names.
        """
        return list(self._options.keys())

    def describe_option(self, pat: str) -> str:
        """Describe the specified option.

        Args:
            pat: The option name.

        Returns:
            A string describing the option and its current value.

        Raises:
            ValueError: If the option name is unknown.
        """
        if pat in self._descriptions:
            return f"{pat}: {self._descriptions[pat]} (current value: {self._options[pat]})"

        msg = f"Unknown option: {pat}"
        raise ValueError(msg)

    @staticmethod
    def flatten_options(k: str, v: OptionTypes, parent_key: str = "") -> dict[str, OptionTypes]:
        """Flatten nested options into a single dictionary."""
        if parent_key != "":
            parent_key += "."

        if isinstance(v, dict):
            ret_dict = {}
            for sub_key, sub_value in v.items():
                ret_dict.update(Options.flatten_options(sub_key, sub_value, parent_key=f"{parent_key}{k}"))
            return ret_dict

        return {f"{parent_key}{k}": v}

    @classmethod
    def load_from_project(cls) -> "Options":
        """Try to load options from a pyretailscience.toml file in the project root directory.

        If the project root directory cannot be found, return a default Options instance.

        Returns:
            An Options instance with options loaded from the pyretailscience.toml file or default
        """
        options_instance = cls()

        project_root = find_project_root()
        if project_root is None:
            return options_instance

        toml_file = Path(project_root) / "pyretailscience.toml"
        if toml_file.is_file():
            return Options.load_from_toml(toml_file)

        return options_instance

    @classmethod
    def load_from_toml(cls, file_path: str | None = None) -> "Options":
        """Load options from a TOML file.

        Args:
            file_path: The path to the TOML file.

        Raises:
            ValueError: If the TOML file contains unknown options.
        """
        options_instance = cls()

        with open(file_path) as f:
            toml_data = toml.load(f)

        for section, options in toml_data.items():
            for option_name, option_value in Options.flatten_options(section, options).items():
                if option_name in options_instance._options:  # noqa: SLF001
                    options_instance.set_option(option_name, option_value)
                else:
                    msg = f"Unknown option in TOML file: {option_name}"
                    raise ValueError(msg)

        return options_instance


@lru_cache
def find_project_root() -> str | None:
    """Returns the directory containing .git, .hg, or pyproject.toml, starting from the current working directory."""
    current_dir = Path.cwd()

    while True:
        if (Path(current_dir / ".git")).is_dir() or (Path(current_dir / "pyretailscience.toml")).is_file():
            return current_dir

        parent_dir = Path(current_dir).parent
        reached_root = parent_dir == current_dir
        if reached_root:
            return None

        current_dir = parent_dir


# Global instance of Options
_global_options = Options().load_from_project()


def set_option(pat: str, val: OptionTypes) -> None:
    """Set the value of the specified option.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.
        val: The value to set the option to.

    Raises:
        ValueError: If the option name is unknown.
    """
    _global_options.set_option(pat, val)


def get_option(pat: str) -> OptionTypes:
    """Get the value of the specified option.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.

    Returns:
        The value of the option.

    Raises:
        ValueError: If the option name is unknown.
    """
    return _global_options.get_option(pat)


def reset_option(pat: str) -> None:
    """Reset the specified option to its default value.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.

    Raises:
        ValueError: If the option name is unknown.
    """
    _global_options.reset_option(pat)


def list_options() -> list[str]:
    """List all available options.

    This is a global function that delegates to the _global_options instance.

    Returns:
        A list of all option names.
    """
    return _global_options.list_options()


def describe_option(pat: str) -> str:
    """Describe the specified option.

    This is a global function that delegates to the _global_options instance.

    Args:
        pat: The option name.

    Returns:
        A string describing the option and its current value.

    Raises:
        ValueError: If the option name is unknown.
    """
    return _global_options.describe_option(pat)


@contextmanager
def option_context(*args: OptionTypes) -> Generator[None, None, None]:
    """Context manager to temporarily set options.

    Temporarily set options and restore them to their previous values after the
    context exits. The arguments should be supplied as alternating option names
    and values.

    Args:
        *args: An even number of arguments, alternating between option names (str)
               and their corresponding values.

    Yields:
        None

    Raises:
        ValueError: If an odd number of arguments is supplied.

    Example:
        >>> with option_context('display.max_rows', 10, 'display.max_columns', 5):
        ...     # Do something with modified options
        ...     pass
        >>> # Options are restored to their previous values here
    """
    if len(args) % 2 != 0:
        raise ValueError("The context manager requires an even number of arguments")

    old_options: dict[str, OptionTypes] = {}
    try:
        for pat, val in zip(args[::2], args[1::2], strict=True):
            old_options[pat] = get_option(pat)
            set_option(pat, val)
        yield
    finally:
        for pat, val in old_options.items():
            set_option(pat, val)
