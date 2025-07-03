#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apicultor: Music Information Retrieval and Sound Analysis Toolkit

A modular toolkit for music information retrieval, sound analysis, and algorithmic composition.
"""

from .config import get_settings
from .core import get_plugin_manager, PluginManager

__version__ = "0.2.0"
__author__ = "Hernán Ordiales, Marcelo Tuller"
__email__ = "hordiales@gmail.com"
__license__ = "GPLv3"

# Initialize plugin system on import
def initialize():
    """Initialize apicultor with plugin system."""
    settings = get_settings()
    plugin_manager = get_plugin_manager()
    
    # Configure plugin manager with settings
    if hasattr(settings, 'plugins'):
        plugin_manager.config.enabled_modules = settings.plugins.enabled_modules
        plugin_manager.config.disabled_modules = settings.plugins.disabled_modules
        plugin_manager.config.auto_discover = settings.plugins.auto_discover
        plugin_manager.config.fail_on_plugin_error = settings.plugins.fail_on_plugin_error
        plugin_manager.config.plugin_configs = settings.plugins.plugin_configs
    
    return plugin_manager

# Lazy initialization
_initialized = False

def get_initialized_manager():
    """Get the initialized plugin manager."""
    global _initialized
    if not _initialized:
        initialize()
        _initialized = True
    return get_plugin_manager()

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "get_settings",
    "get_plugin_manager",
    "PluginManager",
    "initialize",
    "get_initialized_manager",
]