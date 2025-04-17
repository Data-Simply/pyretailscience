"""Manages plugins for pyretailscience package."""

# flake8: noqa: ERA001 ANN001 ANN002 ANN003 ANN204 BLE001 ARG001 ANN201 ANN202 UP008
import functools
import types
from collections.abc import Callable
from importlib.metadata import entry_points


class PluginManager:
    """Manages plugins for pyretailscience package."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for plugin manager."""
        # print(f"==>> cls: {cls}")
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance._classes = {}  # Store methods by class
            cls._instance._registered_classes = set()  # Track registered classes
            cls._instance._plugin_loaded = False  # Track if plugins are loaded
        return cls._instance

    def register_method(self, target_class, name: str, method: Callable) -> None:
        """Register a method with the plugin manager for a specific class.

        Args:
            target_class: Class to extend
            name: Method name
            method: Method to register
        """
        class_name = target_class.__name__
        if class_name not in self._classes:
            self._classes[class_name] = {}
        self._classes[class_name][name] = method

    def extensible(self, cls):
        """Class decorator to make a class extensible via plugins."""
        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Load plugins on first initialization if not loaded yet
            plugin_mgr = plugin_manager  # Get the singleton instance
            if not plugin_mgr._plugin_loaded:
                plugin_mgr._load_all_plugins()
            # Apply plugins to this instance
            plugin_mgr.apply_methods_to_instance(self)

        cls.__init__ = wrapped_init
        self._registered_classes.add(cls.__name__)
        return cls

    def _load_all_plugins(self):
        """Load all plugins from entry points once."""
        try:
            for entry_point in entry_points(group="pyretailscience.plugins"):
                plugin_register_fn = entry_point.load()
                plugin_register_fn(self)
            self._plugin_loaded = True
        except Exception:
            # print(f"Error loading plugins: {e}")
            import traceback

            traceback.print_exc()

    def apply_methods_to_instance(self, instance):
        """Apply all registered methods to an instance."""
        class_name = instance.__class__.__name__
        if class_name in self._classes:
            for name, method in self._classes[class_name].items():
                # Create a bound method and attach it directly to the instance
                # We need to capture the method in a closure
                def create_method(method_func):
                    def bound_method(self, *args, **kwargs):
                        return method_func(*args, **kwargs)

                    return bound_method

                bound = types.MethodType(create_method(method), instance)
                setattr(instance, name, bound)

    def discover_plugins(self, instance) -> None:
        """Discover and load plugins from entry points.

        Args:
            instance: Instance to extend with plugin methods
        """
        # print("in discover_plugins")
        # print(f"==>> instance: {instance}")
        try:
            for entry_point in entry_points(group="pyretailscience.plugins"):
                plugin_register_fn = entry_point.load()
                plugin_register_fn(self, instance)

            # Apply all registered methods for this class to the instance
            self.apply_methods_to_instance(instance)
        except Exception:
            # print(f"Error loading plugins: {e}")
            import traceback

            traceback.print_exc()


# Global plugin manager instance
plugin_manager = PluginManager()
