"""Test cases for plugin system."""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from apicultor.core.plugin_manager import PluginManager, PluginConfig
from apicultor.core.interfaces import PluginInterface, DatabaseInterface
from apicultor.plugins.database_plugin import DatabasePlugin
from apicultor.plugins.constraints_plugin import ConstraintsPlugin


class MockPlugin(PluginInterface):
    """Mock plugin for testing."""
    
    def __init__(self):
        self._enabled = False
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "mock_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Mock plugin for testing"
    
    @property
    def dependencies(self) -> list:
        return []
    
    def initialize(self, config):
        self._initialized = True
        self._enabled = True
    
    def shutdown(self):
        self._enabled = False
        self._initialized = False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled


class TestPluginConfig:
    """Test plugin configuration."""
    
    def test_default_values(self):
        """Test default plugin configuration values."""
        config = PluginConfig()
        assert "database" in config.enabled_modules
        assert "emotion" in config.enabled_modules
        assert len(config.disabled_modules) == 0
        assert config.auto_discover is True
        assert config.fail_on_plugin_error is False
    
    def test_validation_success(self):
        """Test successful plugin configuration validation."""
        config = PluginConfig()
        config.validate()  # Should not raise
    
    def test_validation_conflict(self):
        """Test validation failure with conflicting enabled/disabled modules."""
        config = PluginConfig()
        config.enabled_modules = ["database", "emotion"]
        config.disabled_modules = ["database"]
        
        with pytest.raises(ValueError, match="cannot be both enabled and disabled"):
            config.validate()
    
    def test_plugin_config_validation(self):
        """Test validation of individual plugin configurations."""
        config = PluginConfig()
        config.plugin_configs = {"test_plugin": "not_a_dict"}
        
        with pytest.raises(ValueError, match="must be a dictionary"):
            config.validate()


class TestPluginManager:
    """Test plugin manager functionality."""
    
    def test_initialization(self):
        """Test plugin manager initialization."""
        config = PluginConfig()
        config.enabled_modules = []  # Start with no modules
        config.auto_discover = False
        
        manager = PluginManager(config)
        manager.initialize()
        
        assert manager._initialized is True
        assert len(manager.list_enabled_plugins()) == 0
    
    def test_plugin_registration(self):
        """Test manual plugin registration."""
        config = PluginConfig()
        config.auto_discover = False
        
        manager = PluginManager(config)
        
        # Manually register a plugin class
        manager._plugin_classes["mock"] = MockPlugin
        
        # Enable the plugin
        success = manager.enable_plugin("mock")
        assert success is True
        assert manager.is_plugin_enabled("mock")
        assert len(manager.list_enabled_plugins()) == 1
    
    def test_plugin_enable_disable(self):
        """Test enabling and disabling plugins."""
        config = PluginConfig()
        config.auto_discover = False
        
        manager = PluginManager(config)
        manager._plugin_classes["mock"] = MockPlugin
        
        # Enable plugin
        assert manager.enable_plugin("mock") is True
        assert manager.is_plugin_enabled("mock")
        
        # Get plugin instance
        plugin = manager.get_plugin("mock")
        assert plugin is not None
        assert plugin.is_enabled is True
        
        # Disable plugin
        assert manager.disable_plugin("mock") is True
        assert not manager.is_plugin_enabled("mock")
        assert manager.get_plugin("mock") is None
    
    def test_plugin_dependencies(self):
        """Test plugin dependency checking."""
        class DependentPlugin(PluginInterface):
            @property
            def name(self):
                return "dependent"
            
            @property
            def version(self):
                return "1.0.0"
            
            @property
            def description(self):
                return "Plugin with dependencies"
            
            @property
            def dependencies(self):
                return ["mock_plugin"]
            
            def initialize(self, config):
                pass
            
            def shutdown(self):
                pass
            
            @property
            def is_enabled(self):
                return True
        
        config = PluginConfig()
        config.auto_discover = False
        
        manager = PluginManager(config)
        manager._plugin_classes["dependent"] = DependentPlugin
        
        # Try to enable plugin without dependencies
        success = manager.enable_plugin("dependent")
        assert success is False  # Should fail due to missing dependency
    
    def test_plugin_reload(self):
        """Test plugin reloading."""
        config = PluginConfig()
        config.auto_discover = False
        
        manager = PluginManager(config)
        manager._plugin_classes["mock"] = MockPlugin
        
        # Enable plugin
        manager.enable_plugin("mock")
        assert manager.is_plugin_enabled("mock")
        
        # Reload plugin
        success = manager.reload_plugin("mock")
        assert success is True
        assert manager.is_plugin_enabled("mock")
    
    def test_get_plugins_by_interface(self):
        """Test getting plugins by interface type."""
        config = PluginConfig()
        config.auto_discover = False
        
        manager = PluginManager(config)
        manager._plugin_classes["mock"] = MockPlugin
        
        manager.enable_plugin("mock")
        
        # Get plugins by interface
        plugins = manager.get_plugins_by_interface(PluginInterface)
        assert len(plugins) == 1
        assert plugins[0].name == "mock_plugin"


class TestDatabasePlugin:
    """Test database plugin implementation."""
    
    def test_plugin_properties(self):
        """Test database plugin properties."""
        plugin = DatabasePlugin()
        assert plugin.name == "database"
        assert plugin.version == "1.0.0"
        assert len(plugin.dependencies) == 0
        assert plugin.is_enabled is False
    
    def test_initialization(self):
        """Test database plugin initialization."""
        plugin = DatabasePlugin()
        config = {
            "default_provider": "local",
            "data_dir": "./test_data"
        }
        
        # Mock the database providers to avoid external dependencies
        with patch('apicultor.plugins.database_plugin.FreesoundDB'), \
             patch('apicultor.plugins.database_plugin.RedPanalDB'), \
             patch('apicultor.plugins.database_plugin.JsonMirFilesData'):
            
            plugin.initialize(config)
            assert plugin.is_enabled is True
            assert len(plugin.list_providers()) > 0
    
    def test_search_sounds(self):
        """Test sound search functionality."""
        plugin = DatabasePlugin()
        
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.search.return_value = [{"id": "test_sound", "name": "Test Sound"}]
        
        plugin._providers = {"test": mock_provider}
        plugin._default_provider = "test"
        plugin._enabled = True
        
        results = plugin.search_sounds("test query", limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "test_sound"
        
        mock_provider.search.assert_called_once_with("test query", None, 5)
    
    def test_get_sound_by_id(self):
        """Test getting sound by ID."""
        plugin = DatabasePlugin()
        
        # Mock provider
        mock_provider = MagicMock()
        mock_provider.get_sound.return_value = {"id": "123", "name": "Test Sound"}
        
        plugin._providers = {"test": mock_provider}
        plugin._default_provider = "test"
        plugin._enabled = True
        
        result = plugin.get_sound_by_id("123")
        assert result is not None
        assert result["id"] == "123"
        
        mock_provider.get_sound.assert_called_once_with("123")
    
    def test_shutdown(self):
        """Test database plugin shutdown."""
        plugin = DatabasePlugin()
        
        # Mock provider with close method
        mock_provider = MagicMock()
        plugin._providers = {"test": mock_provider}
        plugin._enabled = True
        
        plugin.shutdown()
        assert plugin.is_enabled is False
        assert len(plugin._providers) == 0


class TestConstraintsPlugin:
    """Test constraints plugin implementation."""
    
    def test_plugin_properties(self):
        """Test constraints plugin properties."""
        plugin = ConstraintsPlugin()
        assert plugin.name == "constraints"
        assert plugin.version == "1.0.0"
        assert len(plugin.dependencies) == 0
        assert plugin.is_enabled is False
    
    def test_initialization(self):
        """Test constraints plugin initialization."""
        plugin = ConstraintsPlugin()
        config = {
            "tolerance": 1e-8,
            "max_iterations": 500
        }
        
        plugin.initialize(config)
        assert plugin.is_enabled is True
        assert plugin._tolerance == 1e-8
        assert plugin._max_iterations == 500
    
    def test_initialization_validation(self):
        """Test constraints plugin initialization validation."""
        plugin = ConstraintsPlugin()
        
        # Invalid tolerance
        with pytest.raises(ValueError, match="Tolerance must be positive"):
            plugin.initialize({"tolerance": -1})
        
        # Invalid max iterations
        with pytest.raises(ValueError, match="Max iterations must be positive"):
            plugin.initialize({"max_iterations": 0})
    
    def test_constraint_methods_require_initialization(self):
        """Test that constraint methods require plugin to be initialized."""
        plugin = ConstraintsPlugin()
        
        import numpy as np
        test_array = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(RuntimeError, match="not initialized"):
            plugin.apply_lower_bounds(test_array)
    
    def test_get_constraint_info(self):
        """Test getting constraint information."""
        plugin = ConstraintsPlugin()
        plugin.initialize({})
        
        info = plugin.get_constraint_info()
        assert "tolerance" in info
        assert "max_iterations" in info
        assert "enabled" in info
        assert "available_methods" in info
        assert info["enabled"] is True


class TestPluginSystemIntegration:
    """Integration tests for the plugin system."""
    
    @patch.dict(os.environ, {
        "APICULTOR_ENABLED_MODULES": "database,constraints",
        "APICULTOR_DISABLED_MODULES": "emotion",
        "APICULTOR_AUTO_DISCOVER": "true"
    })
    def test_environment_configuration(self):
        """Test plugin configuration from environment variables."""
        from apicultor.config.settings import Settings
        
        settings = Settings()
        assert "database" in settings.plugins.enabled_modules
        assert "constraints" in settings.plugins.enabled_modules
        assert "emotion" in settings.plugins.disabled_modules
        assert settings.plugins.auto_discover is True
    
    def test_plugin_manager_with_real_plugins(self):
        """Test plugin manager with actual plugin implementations."""
        config = PluginConfig()
        config.enabled_modules = ["constraints"]
        config.disabled_modules = []
        config.auto_discover = False
        config.plugin_configs = {
            "constraints": {
                "tolerance": 1e-6,
                "max_iterations": 100
            }
        }
        
        manager = PluginManager(config)
        
        # Manually register the constraints plugin
        manager._plugin_classes["constraints"] = ConstraintsPlugin
        
        # Initialize and enable
        manager.initialize()
        
        # Check that plugin is enabled
        assert manager.is_plugin_enabled("constraints")
        
        # Get plugin instance
        constraints_plugin = manager.get_plugin("constraints")
        assert constraints_plugin is not None
        assert isinstance(constraints_plugin, ConstraintsPlugin)
        assert constraints_plugin.is_enabled
        
        # Test plugin functionality
        info = constraints_plugin.get_constraint_info()
        assert info["tolerance"] == 1e-6
        assert info["max_iterations"] == 100
    
    def test_configuration_file_loading(self):
        """Test loading plugin configuration from file."""
        config_data = {
            "plugins": {
                "enabled_modules": ["database", "constraints"],
                "disabled_modules": ["emotion"],
                "plugin_configs": {
                    "database": {
                        "default_provider": "freesound"
                    },
                    "constraints": {
                        "tolerance": 1e-9
                    }
                }
            }
        }
        
        from apicultor.config.settings import Settings
        import json
        from unittest.mock import mock_open
        
        with patch("builtins.open", mock_open(read_data=json.dumps(config_data))), \
             patch("pathlib.Path.exists", return_value=True):
            
            settings = Settings(Path("test_config.json"))
            
            assert "database" in settings.plugins.enabled_modules
            assert "constraints" in settings.plugins.enabled_modules
            assert "emotion" in settings.plugins.disabled_modules
            assert settings.plugins.plugin_configs["database"]["default_provider"] == "freesound"
            assert settings.plugins.plugin_configs["constraints"]["tolerance"] == 1e-9