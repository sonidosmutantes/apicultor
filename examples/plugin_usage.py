#!/usr/bin/env python3
"""
Example usage of the Apicultor plugin system.

This example demonstrates how to:
1. Configure and initialize the plugin system
2. Enable/disable specific modules
3. Use plugin functionality
4. Access different database providers
"""

import os
import numpy as np
from pathlib import Path

# Set up environment for this example
os.environ.update({
    "APICULTOR_ENABLED_MODULES": "database,constraints",
    "APICULTOR_DISABLED_MODULES": "emotion,state_machine",
    "APICULTOR_AUTO_DISCOVER": "true",
    "APICULTOR_LOG_LEVEL": "INFO"
})

import apicultor
from apicultor.config import get_settings
from apicultor.core import get_plugin_manager


def example_basic_usage():
    """Basic plugin system usage."""
    print("=== Basic Plugin System Usage ===")
    
    # Initialize the plugin system
    plugin_manager = apicultor.initialize()
    
    # List available and enabled plugins
    print(f"Available plugins: {plugin_manager.list_available_plugins()}")
    print(f"Enabled plugins: {plugin_manager.list_enabled_plugins()}")
    
    # Check if specific plugins are enabled
    print(f"Database plugin enabled: {plugin_manager.is_plugin_enabled('database')}")
    print(f"Constraints plugin enabled: {plugin_manager.is_plugin_enabled('constraints')}")
    print(f"Emotion plugin enabled: {plugin_manager.is_plugin_enabled('emotion')}")


def example_database_plugin():
    """Example using the database plugin."""
    print("\n=== Database Plugin Usage ===")
    
    plugin_manager = get_plugin_manager()
    
    # Get the database plugin
    db_plugin = plugin_manager.get_plugin('database')
    if not db_plugin:
        print("Database plugin not available")
        return
    
    print(f"Database plugin: {db_plugin.name} v{db_plugin.version}")
    print(f"Description: {db_plugin.description}")
    
    # List available database providers
    providers = db_plugin.list_providers()
    print(f"Available database providers: {providers}")
    
    # Get provider information
    for provider in providers:
        info = db_plugin.get_provider_info(provider)
        print(f"Provider {provider}: {info}")
    
    # Example search (would require actual API keys for real usage)
    try:
        results = db_plugin.search_sounds("piano", limit=3, provider="local")
        print(f"Search results: {len(results)} sounds found")
    except Exception as e:
        print(f"Search failed (expected without real data): {e}")


def example_constraints_plugin():
    """Example using the constraints plugin."""
    print("\n=== Constraints Plugin Usage ===")
    
    plugin_manager = get_plugin_manager()
    
    # Get the constraints plugin
    constraints_plugin = plugin_manager.get_plugin('constraints')
    if not constraints_plugin:
        print("Constraints plugin not available")
        return
    
    print(f"Constraints plugin: {constraints_plugin.name} v{constraints_plugin.version}")
    
    # Get constraint information
    info = constraints_plugin.get_constraint_info()
    print(f"Constraint settings: {info}")
    
    # Example constraint operations
    test_values = np.array([-1.0, 0.5, 2.0, -0.3, 1.5])
    print(f"Original values: {test_values}")
    
    # Apply lower bounds (remove negative values)
    lower_bounded = constraints_plugin.apply_lower_bounds(test_values.copy())
    print(f"After lower bounds: {lower_bounded}")
    
    # Apply upper bounds
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    upper_bounded = constraints_plugin.apply_upper_bounds(lower_bounded, weights, 1.0)
    print(f"After upper bounds: {upper_bounded}")
    
    # Validate constraints
    validation = constraints_plugin.validate_constraints(upper_bounded)
    print(f"Constraint validation: {validation}")


def example_dynamic_plugin_management():
    """Example of dynamic plugin management."""
    print("\n=== Dynamic Plugin Management ===")
    
    plugin_manager = get_plugin_manager()
    
    # Show current state
    print(f"Currently enabled: {plugin_manager.list_enabled_plugins()}")
    
    # Try to enable a disabled plugin (if any are available)
    available = plugin_manager.list_available_plugins()
    enabled = set(plugin_manager.list_enabled_plugins())
    disabled = [p for p in available if p not in enabled]
    
    if disabled:
        plugin_to_enable = disabled[0]
        print(f"\nTrying to enable plugin: {plugin_to_enable}")
        
        success = plugin_manager.enable_plugin(plugin_to_enable)
        print(f"Enable success: {success}")
        
        if success:
            print(f"Now enabled: {plugin_manager.list_enabled_plugins()}")
            
            # Disable it again
            plugin_manager.disable_plugin(plugin_to_enable)
            print(f"After disabling: {plugin_manager.list_enabled_plugins()}")
    else:
        print("No disabled plugins available to demonstrate with")


def example_configuration_management():
    """Example of configuration management."""
    print("\n=== Configuration Management ===")
    
    # Get current settings
    settings = get_settings()
    
    print("Current plugin configuration:")
    print(f"  Enabled modules: {settings.plugins.enabled_modules}")
    print(f"  Disabled modules: {settings.plugins.disabled_modules}")
    print(f"  Auto discover: {settings.plugins.auto_discover}")
    print(f"  Fail on error: {settings.plugins.fail_on_plugin_error}")
    
    # Show plugin-specific configurations
    if settings.plugins.plugin_configs:
        print("\nPlugin-specific configurations:")
        for plugin_name, config in settings.plugins.plugin_configs.items():
            print(f"  {plugin_name}: {config}")
    
    # Export configuration
    config_dict = settings.to_dict()
    plugin_config = config_dict.get('plugins', {})
    print(f"\nExported plugin config: {plugin_config}")


def example_configuration_from_file():
    """Example of loading configuration from file."""
    print("\n=== Configuration from File ===")
    
    # Path to the example configuration file
    config_file = Path("apicultor_config.json")
    
    if config_file.exists():
        print(f"Loading configuration from: {config_file}")
        
        # Create new settings instance from file
        from apicultor.config.settings import Settings
        file_settings = Settings(config_file)
        
        print("Plugin configuration from file:")
        print(f"  Enabled modules: {file_settings.plugins.enabled_modules}")
        print(f"  Plugin configs: {list(file_settings.plugins.plugin_configs.keys())}")
        
        # Show specific plugin configurations
        for plugin_name, config in file_settings.plugins.plugin_configs.items():
            print(f"  {plugin_name} config: {config}")
    else:
        print(f"Configuration file not found: {config_file}")
        print("You can create one using the example in apicultor_config.json")


def main():
    """Run all examples."""
    print("Apicultor Plugin System Examples")
    print("=" * 40)
    
    try:
        example_basic_usage()
        example_database_plugin()
        example_constraints_plugin()
        example_dynamic_plugin_management()
        example_configuration_management()
        example_configuration_from_file()
        
        print("\n=== Examples completed successfully! ===")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        plugin_manager = get_plugin_manager()
        plugin_manager.shutdown()
        print("\nPlugin system shutdown complete.")


if __name__ == "__main__":
    main()