"""Plugin management module for pyretailscience."""

import functools
import inspect
import traceback
import types
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from typing import Optional, Union


class PluginManager:
    """Manages plugins for pyretailscience package."""

    _instance: Optional["PluginManager"] = None

    def __new__(cls: type["PluginManager"]) -> "PluginManager":
        """Singleton pattern for plugin manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._classes = {}
            cls._instance._functions = {}
            cls._instance._registered_classes = set()
            cls._instance._registered_functions = set()
            cls._instance._plugin_loaded = False
        return cls._instance

    def register_method(self, target_class: type, name: str, method: Callable) -> None:
        """Register a method with the plugin manager for a specific class.

        Args:
            target_class: Class to extend
            name: Method name
            method: Method to register
        """
        class_name = target_class.__name__
        self._classes.setdefault(class_name, {})[name] = method

    def register_function_extension(self, target_function: Callable, name: str, method: Callable) -> None:
        """Register an extension method for a standalone function.

        Args:
            target_function: Function to extend
            name: Extension method name
            method: Method to register
        """
        function_name = target_function.__name__
        self._functions.setdefault(function_name, {})[name] = method

    def extensible(self, cls_or_func: type | Callable) -> type | Callable:
        """Decorator to make a class or function extensible via plugins."""
        if inspect.isclass(cls_or_func):
            return self._make_class_extensible(cls_or_func)
        return self._make_function_extensible(cls_or_func)

    def _make_class_extensible(self, cls: type) -> type:
        """Make a class extensible via plugins."""
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            original_init(self, *args, **kwargs)
            plugin_mgr = plugin_manager
            if not plugin_mgr._plugin_loaded:
                plugin_mgr._load_all_plugins()
            plugin_mgr.apply_methods_to_instance(self)

        cls.__init__ = wrapped_init
        self._registered_classes.add(cls.__name__)
        return cls

    def _make_function_extensible(self, func: Callable) -> Callable:
        """Make a function extensible via plugins."""

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            if not plugin_manager._plugin_loaded:
                plugin_manager._load_all_plugins()

            result = func(*args, **kwargs)

            function_name = func.__name__
            if function_name in self._functions:
                return ExtensibleFunctionResult(result, function_name)
            return result

        self._registered_functions.add(func.__name__)
        return wrapped_func

    def _load_all_plugins(self) -> None:
        """Load all plugins from entry points once."""
        failed_plugins = []
        try:
            for entry_point in entry_points(group="pyretailscience.plugins"):
                try:
                    plugin_register_fn = entry_point.load()
                    plugin_register_fn(self)
                except Exception as plugin_error:  # noqa: BLE001
                    failed_plugins.append((entry_point.name, str(plugin_error)))
                    traceback.print_exc()
            self._plugin_loaded = True
            if failed_plugins:
                print(f"Warning: {len(failed_plugins)} plugins failed to load: {failed_plugins}")  # noqa: T201
        except Exception as e:  # noqa: BLE001
            print(f"Error loading plugins: {e}")  # noqa: T201
            traceback.print_exc()

    def apply_methods_to_instance(self, instance: object) -> None:
        """Apply all registered methods to an instance."""
        class_name = instance.__class__.__name__
        methods = self._classes.get(class_name, {})
        for name, method in methods.items():
            bound = types.MethodType(method, instance)
            setattr(instance, name, bound)

    def apply_extensions_to_function_result(self, wrapper: object, function_name: str) -> None:
        """Apply registered extensions to a function result wrapper."""
        for name, method in self._functions.get(function_name, {}).items():

            def create_extension(wrap: object, extension_method: Callable[..., None], method_name: str) -> None:
                def extension_wrapper(*args: tuple, **kwargs: dict) -> object:
                    return extension_method(wrap._result, *args, **kwargs)

                extension_wrapper.__name__ = method_name
                return extension_wrapper

            extension = create_extension(wrapper, method, name)
            setattr(wrapper, name, extension)


class ExtensibleFunctionResult:
    """Wrapper for function results that can have extension methods."""

    def __init__(self, result: object, function_name: str) -> None:
        """Initialize a FunctionResultWrapper."""
        self._result = result
        self._function_name = function_name
        plugin_manager.apply_extensions_to_function_result(self, function_name)

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the wrapped result."""
        if hasattr(self._result, name):
            return getattr(self._result, name)
        type_name = type(self._result).__name__
        message = f"'{type_name}' object has no attribute '{name}'"
        raise AttributeError(message)

    def __repr__(self) -> str:
        """Represent the wrapper as the wrapped result."""
        return repr(self._result)

    def __str__(self) -> str:
        """Convert the wrapper to string as the wrapped result."""
        return str(self._result)

    def __iter__(self) -> Union[None, "Iterator"]:
        """Return an iterator if the result supports iteration."""
        if hasattr(self._result, "__iter__"):
            return iter(self._result)
        type_name = type(self._result).__name__
        message = f"'{type_name}' object is not iterable"
        raise TypeError(message)

    def __getitem__(self, key: object) -> object:
        """Support indexing if the wrapped result supports it."""
        if not hasattr(self._result, "__getitem__"):
            type_name = type(self._result).__name__
            message = f"'{type_name}' object does not support indexing"
            raise TypeError(message)
        return self._result[key]

    def __len__(self) -> int:
        """Return the length of the wrapped result if it has one."""
        if not hasattr(self._result, "__len__"):
            type_name = type(self._result).__name__
            message = f"'{type_name}' object has no len()"
            raise TypeError(message)
        return len(self._result)

    def __add__(self, other: object) -> object:
        """Support addition if the wrapped result supports it."""
        if isinstance(other, ExtensibleFunctionResult):
            return self._result + other._result
        return self._result + other

    def __radd__(self, other: object) -> object:
        """Support right addition if the wrapped result supports it."""
        return other + self._result

    def __mul__(self, other: object) -> object:
        """Support multiplication if the wrapped result supports it."""
        return self._result * other

    def __rmul__(self, other: object) -> object:
        """Support right multiplication if the wrapped result supports it."""
        return other * self._result

    def __eq__(self, other: object) -> bool:
        """Support equality comparison."""
        if isinstance(other, ExtensibleFunctionResult):
            return self._result == other._result
        return self._result == other

    def __lt__(self, other: object) -> bool:
        """Support less than comparison."""
        if isinstance(other, ExtensibleFunctionResult):
            return self._result < other._result
        return self._result < other

    def __gt__(self, other: object) -> bool:
        """Support greater than comparison."""
        if isinstance(other, ExtensibleFunctionResult):
            return self._result > other._result
        return self._result > other

    def __le__(self, other: object) -> bool:
        """Support less than or equal comparison."""
        if isinstance(other, ExtensibleFunctionResult):
            return self._result <= other._result
        return self._result <= other

    def __ge__(self, other: object) -> bool:
        """Support greater than or equal comparison."""
        if isinstance(other, ExtensibleFunctionResult):
            return self._result >= other._result
        return self._result >= other


plugin_manager = PluginManager()
