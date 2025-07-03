"""Core apicultor functionality and plugin system."""

from .plugin_manager import PluginManager, get_plugin_manager
from .interfaces import PluginInterface, AudioProcessorInterface, DatabaseInterface

__all__ = [
    "PluginManager",
    "get_plugin_manager", 
    "PluginInterface",
    "AudioProcessorInterface",
    "DatabaseInterface",
]