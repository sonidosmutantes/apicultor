"""Plugin management system for apicultor."""

import importlib
import importlib.util
import logging
from typing import Dict, List, Optional, Type, Any, Set
from pathlib import Path
import inspect

from .interfaces import PluginInterface
from ..config.base import BaseConfig


logger = logging.getLogger(__name__)


class PluginConfig(BaseConfig):
    """Configuration for plugin management."""
    
    def __init__(self):
        self.enabled_plugins: List[str] = []
        self.disabled_plugins: List[str] = []
        self.plugin_paths: List[str] = []
        self.auto_discover: bool = True
        self.fail_on_plugin_error: bool = False
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def validate(self) -> None:
        """Validate plugin configuration."""
        # Check for conflicts between enabled and disabled lists
        enabled_set = set(self.enabled_plugins)
        disabled_set = set(self.disabled_plugins)
        conflicts = enabled_set & disabled_set
        if conflicts:
            raise ValueError(f"Plugins cannot be both enabled and disabled: {conflicts}")


class PluginManager:
    """Manages loading, enabling, and disabling of apicultor plugins."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize plugin manager.
        
        Args:
            config: Plugin configuration
        """
        self.config = config or PluginConfig()
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_classes: Dict[str, Type[PluginInterface]] = {}
        self._enabled_plugins: Set[str] = set()
        self._initialized = False
        
        # Built-in plugin modules mapping
        self._builtin_modules = {
            "database": "apicultor.plugins.database_plugin",
            "constraints": "apicultor.plugins.constraints_plugin",
        }
    
    def initialize(self) -> None:
        """Initialize the plugin system."""
        if self._initialized:
            return
            
        logger.info("Initializing plugin manager")
        
        # Discover and register plugins
        self._discover_plugins()
        
        # Enable configured plugins
        self._enable_configured_plugins()
        
        self._initialized = True
        logger.info(f"Plugin manager initialized with {len(self._enabled_plugins)} enabled plugins")
    
    def _discover_plugins(self) -> None:
        """Discover available plugins."""
        if not self.config.auto_discover:
            return
            
        # Discover built-in plugins
        for plugin_name, module_path in self._builtin_modules.items():
            try:
                self._discover_plugin_in_module(plugin_name, module_path)
            except Exception as e:
                logger.debug(f"Plugin {plugin_name} not available: {e}")
                if self.config.fail_on_plugin_error:
                    raise
        
        # Discover plugins in additional paths
        for plugin_path in self.config.plugin_paths:
            try:
                self._discover_plugins_in_path(plugin_path)
            except Exception as e:
                logger.warning(f"Failed to discover plugins in {plugin_path}: {e}")
                if self.config.fail_on_plugin_error:
                    raise
    
    def _discover_plugin_in_module(self, plugin_name: str, module_path: str) -> None:
        """Discover plugin classes in a specific module.
        
        Args:
            plugin_name: Name to assign to the plugin
            module_path: Python module path
        """
        try:
            module = importlib.import_module(module_path)
            
            # Look for classes that implement PluginInterface
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj != PluginInterface and
                    hasattr(obj, 'name')):
                    
                    # Use the plugin's declared name or fall back to our name
                    try:
                        actual_name = obj.name if hasattr(obj, 'name') else plugin_name
                        self._plugin_classes[actual_name] = obj
                        logger.debug(f"Discovered plugin class: {actual_name}")
                    except Exception:
                        # If we can't get the name, use our fallback
                        self._plugin_classes[plugin_name] = obj
                        logger.debug(f"Discovered plugin class: {plugin_name}")
                        
        except ImportError as e:
            logger.debug(f"Module {module_path} not available: {e}")
        except Exception as e:
            logger.warning(f"Error discovering plugin in {module_path}: {e}")
            if self.config.fail_on_plugin_error:
                raise
    
    def _discover_plugins_in_path(self, plugin_path: str) -> None:
        """Discover plugins in a filesystem path.
        
        Args:
            plugin_path: Path to search for plugins
        """
        path = Path(plugin_path)
        if not path.exists():
            logger.warning(f"Plugin path does not exist: {plugin_path}")
            return
        
        # Look for Python files that might contain plugins
        for py_file in path.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
                
            try:
                # Convert file path to module path
                relative_path = py_file.relative_to(path)
                module_name = str(relative_path.with_suffix("")).replace("/", ".")
                
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for plugin classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, PluginInterface) and 
                            obj != PluginInterface):
                            plugin_name = getattr(obj, 'name', name.lower())
                            self._plugin_classes[plugin_name] = obj
                            logger.debug(f"Discovered external plugin: {plugin_name}")
                            
            except Exception as e:
                logger.warning(f"Error loading plugin from {py_file}: {e}")
                if self.config.fail_on_plugin_error:
                    raise
    
    def _enable_configured_plugins(self) -> None:
        """Enable plugins based on configuration."""
        # If specific plugins are configured to be enabled, use that list
        if self.config.enabled_plugins:
            plugins_to_enable = self.config.enabled_plugins
        else:
            # Otherwise enable all discovered plugins except those explicitly disabled
            plugins_to_enable = [
                name for name in self._plugin_classes.keys()
                if name not in self.config.disabled_plugins
            ]
        
        for plugin_name in plugins_to_enable:
            if plugin_name not in self.config.disabled_plugins:
                try:
                    self.enable_plugin(plugin_name)
                except Exception as e:
                    logger.debug(f"Plugin {plugin_name} not available: {e}")
                    if self.config.fail_on_plugin_error:
                        raise
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a specific plugin.
        
        Args:
            plugin_name: Name of plugin to enable
            
        Returns:
            True if plugin was enabled successfully
        """
        if plugin_name in self._enabled_plugins:
            logger.debug(f"Plugin {plugin_name} already enabled")
            return True
        
        if plugin_name not in self._plugin_classes:
            logger.debug(f"Plugin {plugin_name} not found")
            return False
        
        try:
            # Instantiate the plugin
            plugin_class = self._plugin_classes[plugin_name]
            plugin_instance = plugin_class()
            
            # Get plugin-specific configuration
            plugin_config = self.config.plugin_configs.get(plugin_name, {})
            
            # Check dependencies
            if not self._check_dependencies(plugin_instance):
                logger.error(f"Dependencies not satisfied for plugin {plugin_name}")
                return False
            
            # Initialize the plugin
            plugin_instance.initialize(plugin_config)
            
            # Store the instance
            self._plugins[plugin_name] = plugin_instance
            self._enabled_plugins.add(plugin_name)
            
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable plugin {plugin_name}: {e}")
            if self.config.fail_on_plugin_error:
                raise
            return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a specific plugin.
        
        Args:
            plugin_name: Name of plugin to disable
            
        Returns:
            True if plugin was disabled successfully
        """
        if plugin_name not in self._enabled_plugins:
            logger.debug(f"Plugin {plugin_name} not enabled")
            return True
        
        try:
            # Get the plugin instance
            plugin_instance = self._plugins[plugin_name]
            
            # Shutdown the plugin
            plugin_instance.shutdown()
            
            # Remove from enabled plugins
            self._enabled_plugins.remove(plugin_name)
            del self._plugins[plugin_name]
            
            logger.info(f"Disabled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable plugin {plugin_name}: {e}")
            return False
    
    def _check_dependencies(self, plugin: PluginInterface) -> bool:
        """Check if plugin dependencies are satisfied.
        
        Args:
            plugin: Plugin instance to check
            
        Returns:
            True if all dependencies are satisfied
        """
        for dependency in plugin.dependencies:
            if dependency not in self._enabled_plugins:
                logger.warning(f"Dependency {dependency} not enabled for plugin {plugin.name}")
                return False
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get an enabled plugin instance.
        
        Args:
            plugin_name: Name of plugin to retrieve
            
        Returns:
            Plugin instance or None if not found/enabled
        """
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_interface(self, interface_type: Type[PluginInterface]) -> List[PluginInterface]:
        """Get all enabled plugins implementing a specific interface.
        
        Args:
            interface_type: Interface class to match
            
        Returns:
            List of plugin instances implementing the interface
        """
        matching_plugins = []
        for plugin in self._plugins.values():
            if isinstance(plugin, interface_type):
                matching_plugins.append(plugin)
        return matching_plugins
    
    def list_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugin names.
        
        Returns:
            List of enabled plugin names
        """
        return list(self._enabled_plugins)
    
    def list_available_plugins(self) -> List[str]:
        """Get list of available plugin names.
        
        Returns:
            List of available plugin names
        """
        return list(self._plugin_classes.keys())
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled.
        
        Args:
            plugin_name: Name of plugin to check
            
        Returns:
            True if plugin is enabled
        """
        return plugin_name in self._enabled_plugins
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (disable and re-enable).
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if plugin was reloaded successfully
        """
        was_enabled = self.is_plugin_enabled(plugin_name)
        
        if was_enabled:
            self.disable_plugin(plugin_name)
        
        # Re-discover the plugin
        if plugin_name in self._builtin_modules:
            module_path = self._builtin_modules[plugin_name]
            self._discover_plugin_in_module(plugin_name, module_path)
        
        if was_enabled:
            return self.enable_plugin(plugin_name)
        
        return True
    
    def shutdown(self) -> None:
        """Shutdown all plugins and cleanup."""
        logger.info("Shutting down plugin manager")
        
        for plugin_name in list(self._enabled_plugins):
            self.disable_plugin(plugin_name)
        
        self._plugins.clear()
        self._plugin_classes.clear()
        self._enabled_plugins.clear()
        self._initialized = False


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager instance.
    
    Returns:
        Global plugin manager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
        _plugin_manager.initialize()
    return _plugin_manager


def initialize_plugin_manager(config: Optional[PluginConfig] = None) -> PluginManager:
    """Initialize the global plugin manager with specific configuration.
    
    Args:
        config: Plugin configuration
        
    Returns:
        Configured plugin manager instance
    """
    global _plugin_manager
    if _plugin_manager is not None:
        _plugin_manager.shutdown()
    
    _plugin_manager = PluginManager(config)
    _plugin_manager.initialize()
    return _plugin_manager