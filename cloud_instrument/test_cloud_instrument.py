#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the modernized Cloud Instrument.

This script tests the Cloud Instrument with minimal dependencies,
demonstrating that it can run even without OSC or SuperCollider components.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_minimal_config():
    """Create a minimal configuration for testing."""
    config = {
        "api": "redpanal",
        "osc": {
            "port": 9001
        },
        "sound": {
            "synth": "none"  # Disable audio synthesis for testing
        },
        "plugins": {
            "enabled_modules": ["database"],
            "disabled_modules": ["analysis", "sonification", "constraints"],
            "plugin_configs": {
                "database": {
                    "default_provider": "redpanal"
                }
            }
        },
        "RedPanal.org": [
            {
                "url": "http://127.0.0.1:5000"
            }
        ],
        "logging": {
            "level": "INFO"
        }
    }
    
    with open("test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return "test_config.json"

def test_cloud_instrument():
    """Test the Cloud Instrument with minimal configuration."""
    from CloudInstrument import CloudInstrument, setup_logging
    
    # Set up logging
    setup_logging()
    
    # Create minimal config
    config_path = create_minimal_config()
    
    try:
        print("=" * 50)
        print("Testing Cloud Instrument with minimal configuration")
        print("=" * 50)
        
        # Create instrument
        instrument = CloudInstrument(config_path)
        
        # Test individual components
        print("\n1. Testing configuration loading...")
        instrument.load_config()
        print("✓ Configuration loaded successfully")
        
        print("\n2. Testing plugin initialization...")
        instrument.initialize_plugins()
        print(f"✓ Plugins initialized: {len(instrument.plugin_manager.list_enabled_plugins())} enabled")
        
        print("\n3. Testing database API setup...")
        instrument.setup_database_api()
        print("✓ Database API setup completed")
        
        print("\n4. Testing OSC server setup...")
        instrument.setup_osc_server()
        if instrument.osc_server:
            print("✓ OSC server started")
        else:
            print("⚠ OSC server not available (expected)")
        
        print("\n5. Testing audio synthesis setup...")
        instrument.setup_audio_synthesis()
        if instrument.audio_server:
            print("✓ Audio synthesis started")
        else:
            print("⚠ Audio synthesis not available (expected)")
        
        print("\n6. Testing component connections...")
        instrument.connect_components()
        print("✓ Component connections completed")
        
        print("\n7. Testing plugin access...")
        database_plugin = instrument.plugin_manager.get_plugin("database")
        if database_plugin:
            print("✓ Database plugin accessible")
        else:
            print("⚠ Database plugin not available")
        
        print("\n8. Testing graceful shutdown...")
        instrument.shutdown()
        print("✓ Shutdown completed successfully")
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED - Cloud Instrument is working!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test config
        if os.path.exists(config_path):
            os.remove(config_path)

def test_plugin_functionality():
    """Test basic plugin functionality."""
    try:
        from apicultor.core.plugin_manager import PluginManager, PluginConfig
        
        print("\n" + "=" * 30)
        print("Testing Plugin System")
        print("=" * 30)
        
        # Create plugin config
        config = PluginConfig()
        config.enabled_plugins = ["database"]
        config.plugin_paths = [str(Path(__file__).parent.parent / "src" / "apicultor" / "plugins")]
        
        # Initialize plugin manager
        manager = PluginManager(config)
        manager.initialize()
        
        print(f"✓ Plugin manager initialized with {len(manager.list_enabled_plugins())} plugins")
        
        # Test database plugin
        db_plugin = manager.get_plugin("database")
        if db_plugin:
            print("✓ Database plugin loaded successfully")
        else:
            print("⚠ Database plugin not found")
        
        # Shutdown
        manager.shutdown()
        print("✓ Plugin system shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Plugin test failed: {e}")
        return False

if __name__ == "__main__":
    print("Cloud Instrument Test Suite")
    print("Testing modern APICultor Cloud Instrument...")
    
    # Test plugin system first
    plugin_test_passed = test_plugin_functionality()
    
    # Test full cloud instrument
    instrument_test_passed = test_cloud_instrument()
    
    print(f"\nTest Results:")
    print(f"Plugin System: {'PASS' if plugin_test_passed else 'FAIL'}")
    print(f"Cloud Instrument: {'PASS' if instrument_test_passed else 'FAIL'}")
    
    if plugin_test_passed and instrument_test_passed:
        print("\n🎉 All tests passed! Cloud Instrument is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1)