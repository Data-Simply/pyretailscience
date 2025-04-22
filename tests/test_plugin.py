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
    original_functions = plugin_manager._functions.copy()
    original_loaded = plugin_manager._plugin_loaded

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

    plugin_manager._functions = original_functions
    plugin_manager._plugin_loaded = original_loaded


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


def test_plugin_manager_load_plugins_no_plugins(monkeypatch):
    """Ensure _load_all_plugins handles the case of no entry points gracefully."""
    monkeypatch.setattr(
        "pyretailscience.plugin.entry_points",
        lambda group=None: [],
    )
    plugin_manager._plugin_loaded = False
    plugin_manager._load_all_plugins()
    assert plugin_manager._plugin_loaded is True


def test_extensible_function_return_value():
    """Test that the extensible function returns the expected result."""
    result = get_number()
    result_output = 42
    assert result == result_output


def test_plugin_load_failure(monkeypatch, capsys):
    """Test that plugin loading errors are caught and logged to stdout."""

    def broken_plugin(pm):
        raise RuntimeError("Boom!")

    monkeypatch.setattr(
        "pyretailscience.plugin.entry_points",
        lambda group=None: [
            types.SimpleNamespace(name="broken_plugin", load=lambda: broken_plugin),
        ],
    )
    plugin_manager._plugin_loaded = False
    plugin_manager._load_all_plugins()
    captured = capsys.readouterr()
    assert "Boom!" in captured.out


def test_len_type_error():
    """Test that calling len() on a non-length compatible ExtensibleFunctionResult raises TypeError."""
    wrapper = ExtensibleFunctionResult(123, "dummy_func")
    with pytest.raises(TypeError, match="object has no len"):
        len(wrapper)


def test_add_type_error():
    """Test that unpacking a non-iterable ExtensibleFunctionResult raises TypeError."""
    wrapper = ExtensibleFunctionResult(123, "dummy_func")
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        _ = [*wrapper, 1, 2]


def test_add_with_other_instance():
    """Test that adding two ExtensibleFunctionResult instances works if both values are compatible."""
    a = ExtensibleFunctionResult("hello ", "dummy_func")
    b = ExtensibleFunctionResult("world", "dummy_func")
    assert a + b == "hello world"


def test_getitem_type_error():
    """Test that indexing on a non-indexable ExtensibleFunctionResult raises TypeError."""
    wrapper = ExtensibleFunctionResult(123, "dummy_func")
    with pytest.raises(TypeError, match="object does not support indexing"):
        wrapper[0]


def test_radd_with_other_instance():
    """Test that right-adding with ExtensibleFunctionResult works if values are compatible."""
    a = "hello "
    b = ExtensibleFunctionResult("world", "dummy_func")
    assert a + b == "hello world"


def test_comparison_with_other_instance():
    """Test comparison operators when comparing two ExtensibleFunctionResult instances."""
    a = ExtensibleFunctionResult(10, "dummy_func")
    b = ExtensibleFunctionResult(20, "dummy_func")
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a != b

    c = ExtensibleFunctionResult(10, "dummy_func")
    assert a == c


def test_bool_conversion():
    """Test boolean conversion of ExtensibleFunctionResult."""
    true_wrapper = ExtensibleFunctionResult([1, 2, 3], "dummy_func")
    false_wrapper = ExtensibleFunctionResult([], "dummy_func")

    assert bool(true_wrapper) is True
    assert bool(false_wrapper) is False

    result1 = "Has items" if true_wrapper else "Empty"
    result2 = "Has items" if false_wrapper else "Empty"

    assert result1 == "Has items"
    assert result2 == "Empty"


def test_nested_attribute_access():
    """Test that nested attribute access works through ExtensibleFunctionResult."""
    number = 42

    class NestedObject:
        def __init__(self):
            self.value = "nested"
            self.number = number

    obj = NestedObject()
    wrapper = ExtensibleFunctionResult(obj, "dummy_func")

    assert wrapper.value == "nested"
    assert wrapper.number == number


def test_make_class_extensible_load_on_demand(monkeypatch):
    """Test that _make_class_extensible loads plugins when needed (line 65)."""
    load_called = {"value": False}

    original_load = plugin_manager._load_all_plugins

    def mock_load():
        load_called["value"] = True
        original_load()

    monkeypatch.setattr(plugin_manager, "_load_all_plugins", mock_load)
    monkeypatch.setattr(plugin_manager, "_plugin_loaded", False)

    MyClass("test")

    assert load_called["value"] is True


def test_make_function_extensible_load_on_demand(monkeypatch):
    """Test that _make_function_extensible loads plugins when needed (line 78)."""
    load_called = {"value": False}

    original_load = plugin_manager._load_all_plugins

    def mock_load():
        load_called["value"] = True
        original_load()

    monkeypatch.setattr(plugin_manager, "_load_all_plugins", mock_load)
    monkeypatch.setattr(plugin_manager, "_plugin_loaded", False)

    @plugin_manager.extensible
    def standalone_func():
        return "no extensions"

    result = standalone_func()

    assert load_called["value"] is True
    assert result == "no extensions"


def test_load_all_plugins_general_exception(monkeypatch, capsys):
    """Test exception handling in _load_all_plugins (lines 104-106)."""

    def mock_entry_points_error(*args, **kwargs):
        raise ValueError("General error in entry_points")

    monkeypatch.setattr("pyretailscience.plugin.entry_points", mock_entry_points_error)

    plugin_manager._plugin_loaded = False

    plugin_manager._load_all_plugins()

    captured = capsys.readouterr()
    assert "Error loading plugins: General error in entry_points" in captured.out


def test_add_with_non_extensible_result():
    """Test __add__ implementation with non-ExtensibleFunctionResult objects (line 185)."""
    wrapper = ExtensibleFunctionResult([1, 2], "dummy_func")

    result = [*wrapper, 3, 4]
    assert result == [1, 2, 3, 4]

    str_wrapper = ExtensibleFunctionResult("Hello", "dummy_func")
    result = str_wrapper + " World"
    assert result == "Hello World"

    num_wrapper = ExtensibleFunctionResult(10, "dummy_func")
    result = num_wrapper + 5
    result_output = 15
    assert result == result_output
