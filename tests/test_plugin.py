"""Tests for the Plugin module."""

import types

import pytest

from pyretailscience.plugin import ExtensibleFunctionResult, PluginManager, plugin_manager


@plugin_manager.extensible
class MyClass:
    """A sample extensible class used to test plugin-based method registration."""

    def __init__(self, x):
        """Initialize MyClass with a value."""
        self.x = x


def test_extensible_class(monkeypatch):
    """Test that a method can be added to an extensible class via plugin."""

    def plugin(pm):
        def greet(self):
            return f"Hello {self.x}"

        pm.register_method(MyClass, "greet", greet)

    monkeypatch.setattr(
        "pyretailscience.plugin.entry_points",
        lambda group=None: [types.SimpleNamespace(load=lambda: plugin)],
    )
    plugin_manager._plugin_loaded = False
    plugin_manager._load_all_plugins()

    obj = MyClass("World")
    assert obj.greet() == "Hello World"


@plugin_manager.extensible
def get_number():
    """A sample extensible function that returns a static number."""
    return 42


def test_extensible_function(monkeypatch):
    """Test that a function extension is registered and callable on the result."""

    def plugin(pm):
        def double(result):
            return result * 2

        pm.register_function_extension(get_number, "double", double)

    monkeypatch.setattr(
        "pyretailscience.plugin.entry_points",
        lambda group=None: [types.SimpleNamespace(load=lambda: plugin)],
    )
    plugin_manager._functions.clear()
    plugin_manager._plugin_loaded = False
    plugin_manager._load_all_plugins()

    result = get_number()
    expected_double = 84
    assert result.double() == expected_double


def test_extensible_result_attr_error():
    """Test that accessing an invalid attribute on ExtensibleFunctionResult raises AttributeError."""
    wrapper = ExtensibleFunctionResult("hello", "dummy_func")
    assert wrapper.upper() == "HELLO"
    with pytest.raises(AttributeError):
        _ = wrapper.no_such_attr


def test_operator_overloads():
    """Test operator overloading (add, multiply, comparison) on ExtensibleFunctionResult."""
    result = ExtensibleFunctionResult([1, 2], "dummy_func")
    assert [*result, 3] == [1, 2, 3]
    assert [0, *result] == [0, 1, 2]
    assert result * 2 == [1, 2, 1, 2]
    assert 2 * result == [1, 2, 1, 2]
    assert result == [1, 2]
    assert result != [3]
    assert result < [2, 2]
    assert result <= [1, 2]
    assert result > [0]
    assert result >= [1]


def test_iter_and_len():
    """Test iteration and length behavior of ExtensibleFunctionResult wrapping an iterable."""
    result = ExtensibleFunctionResult([1, 2, 3], "dummy_func")
    assert list(iter(result)) == [1, 2, 3]
    expected_length = 3
    assert len(result) == expected_length


def test_iter_not_iterable():
    """Test that trying to iterate over a non-iterable ExtensibleFunctionResult raises TypeError."""
    result = ExtensibleFunctionResult(123, "dummy_func")
    with pytest.raises(TypeError):
        list(result)


def test_register_method():
    """Test manual method registration to an extensible class via plugin manager."""

    def custom_method(self):
        return "custom method"

    plugin_manager.register_method(MyClass, "custom_method", custom_method)
    obj = MyClass("Test")
    assert obj.custom_method() == "custom method"


def test_register_function_extension():
    """Test manual function extension registration via plugin manager."""

    def add_five(result):
        return result + 5

    plugin_manager.register_function_extension(get_number, "add_five", add_five)
    result = get_number()
    expected_add_five_result = 47
    assert result.add_five() == expected_add_five_result


def test_load_all_plugins(monkeypatch):
    """Test the full plugin loading flow using entry points to register methods."""

    def plugin(pm):
        def greet(self):
            return f"Hello {self.x}"

        pm.register_method(MyClass, "greet", greet)

    monkeypatch.setattr(
        "pyretailscience.plugin.entry_points",
        lambda group=None: [types.SimpleNamespace(load=lambda: plugin)],
    )

    plugin_manager._plugin_loaded = False
    plugin_manager._load_all_plugins()
    obj = MyClass("World")
    assert obj.greet() == "Hello World"


def test_extensible_function_result_attr_error():
    """Ensure AttributeError is raised for missing attributes in ExtensibleFunctionResult."""
    wrapper = ExtensibleFunctionResult("hello", "dummy_func")
    with pytest.raises(AttributeError):
        _ = wrapper.no_such_attr


def test_plugin_manager_singleton():
    """Test that PluginManager follows the singleton pattern."""
    pm1 = PluginManager()
    pm2 = PluginManager()
    assert pm1 is pm2


def test_extensible_function_result_repr_and_str():
    """Test __repr__ and __str__ of ExtensibleFunctionResult return wrapped value's representation."""
    wrapper = ExtensibleFunctionResult([1, 2, 3], "dummy_func")
    assert repr(wrapper) == "[1, 2, 3]"
    assert str(wrapper) == "[1, 2, 3]"


def test_extensible_function_result_getitem():
    """Test __getitem__ on ExtensibleFunctionResult behaves like wrapped object."""
    wrapper = ExtensibleFunctionResult([1, 2, 3], "dummy_func")
    assert wrapper[0] == 1
