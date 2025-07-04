"""Tests for configuration management."""

import pytest
import json
import tempfile
from pathlib import Path

from ..core.config import (
    CloudInstrumentConfig,
    OSCConfig,
    AudioConfig,
    DatabaseConfig,
    MIDIConfig,
    LoggingConfig,
    AudioBackend,
    DatabaseProvider
)


class TestOSCConfig:
    """Test OSC configuration."""
    
    def test_default_values(self):
        """Test default OSC configuration values."""
        config = OSCConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 9001
        assert config.enable_logging is True
    
    def test_custom_values(self):
        """Test custom OSC configuration values."""
        config = OSCConfig(
            host="0.0.0.0",
            port=9002,
            enable_logging=False
        )
        assert config.host == "0.0.0.0"
        assert config.port == 9002
        assert config.enable_logging is False


class TestAudioConfig:
    """Test audio configuration."""
    
    def test_default_values(self):
        """Test default audio configuration values."""
        config = AudioConfig()
        assert config.backend == AudioBackend.SUPERCOLLIDER
        assert config.sample_rate == 44100
        assert config.buffer_size == 512
        assert config.channels == 2
    
    def test_custom_values(self):
        """Test custom audio configuration values."""
        config = AudioConfig(
            backend=AudioBackend.PYO,
            sample_rate=48000,
            buffer_size=256,
            channels=1
        )
        assert config.backend == AudioBackend.PYO
        assert config.sample_rate == 48000
        assert config.buffer_size == 256
        assert config.channels == 1


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_default_values(self):
        """Test default database configuration values."""
        config = DatabaseConfig()
        assert config.default_provider == DatabaseProvider.FREESOUND
        assert config.freesound_api_key is None
        assert config.redpanal_url == "https://redpanal.org/api/audio"
        assert config.local_data_dir == "./data"
    
    def test_custom_values(self):
        """Test custom database configuration values."""
        config = DatabaseConfig(
            default_provider=DatabaseProvider.LOCAL,
            freesound_api_key="test_key",
            local_data_dir="/custom/data"
        )
        assert config.default_provider == DatabaseProvider.LOCAL
        assert config.freesound_api_key == "test_key"
        assert config.local_data_dir == "/custom/data"


class TestMIDIConfig:
    """Test MIDI configuration."""
    
    def test_default_values(self):
        """Test default MIDI configuration values."""
        config = MIDIConfig()
        assert config.enabled is True
        assert config.input_device is None
        assert config.virtual_port is True
        assert config.client_name == "CloudInstrument"
    
    def test_custom_values(self):
        """Test custom MIDI configuration values."""
        config = MIDIConfig(
            enabled=False,
            input_device="MIDI Device",
            virtual_port=False,
            client_name="CustomClient"
        )
        assert config.enabled is False
        assert config.input_device == "MIDI Device"
        assert config.virtual_port is False
        assert config.client_name == "CustomClient"


class TestCloudInstrumentConfig:
    """Test main configuration class."""
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = CloudInstrumentConfig()
        
        assert isinstance(config.osc, OSCConfig)
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.midi, MIDIConfig)
        assert isinstance(config.logging, LoggingConfig)
        
        assert config.enabled_modules == ["database"]
        assert config.plugin_configs == {}
    
    def test_from_dict_basic(self):
        """Test configuration creation from dictionary."""
        data = {
            "osc": {
                "host": "0.0.0.0",
                "port": 9002
            },
            "audio": {
                "backend": "pyo",
                "sample_rate": 48000
            },
            "enabled_modules": ["database", "analysis"]
        }
        
        config = CloudInstrumentConfig.from_dict(data)
        
        assert config.osc.host == "0.0.0.0"
        assert config.osc.port == 9002
        assert config.audio.backend == AudioBackend.PYO
        assert config.audio.sample_rate == 48000
        assert config.enabled_modules == ["database", "analysis"]
    
    def test_from_dict_with_env_vars(self):
        """Test configuration with environment variable expansion."""
        import os
        
        # Set environment variable
        os.environ["TEST_API_KEY"] = "secret_key_123"
        
        try:
            data = {
                "database": {
                    "freesound_api_key": "${TEST_API_KEY}"
                }
            }
            
            config = CloudInstrumentConfig.from_dict(data)
            assert config.database.freesound_api_key == "secret_key_123"
            
        finally:
            # Clean up
            if "TEST_API_KEY" in os.environ:
                del os.environ["TEST_API_KEY"]
    
    def test_from_dict_invalid_enum(self):
        """Test configuration with invalid enum values."""
        data = {
            "audio": {
                "backend": "invalid_backend"
            }
        }
        
        config = CloudInstrumentConfig.from_dict(data)
        # Should fall back to default
        assert config.audio.backend == AudioBackend.SUPERCOLLIDER
    
    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = CloudInstrumentConfig()
        config.osc.host = "0.0.0.0"
        config.audio.backend = AudioBackend.PYO
        config.enabled_modules = ["database", "analysis"]
        
        data = config.to_dict()
        
        assert data["osc"]["host"] == "0.0.0.0"
        assert data["audio"]["backend"] == "pyo"
        assert data["enabled_modules"] == ["database", "analysis"]
    
    def test_file_operations(self):
        """Test saving and loading configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Create and save configuration
            config = CloudInstrumentConfig()
            config.osc.port = 9002
            config.audio.backend = AudioBackend.PYO
            config.enabled_modules = ["database", "analysis"]
            
            config.save_to_file(config_path)
            
            # Load configuration
            loaded_config = CloudInstrumentConfig.from_file(config_path)
            
            assert loaded_config.osc.port == 9002
            assert loaded_config.audio.backend == AudioBackend.PYO
            assert loaded_config.enabled_modules == ["database", "analysis"]
            
        finally:
            # Clean up
            if config_path.exists():
                config_path.unlink()
    
    def test_file_not_found(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            CloudInstrumentConfig.from_file("non_existent_config.json")
    
    def test_invalid_json(self):
        """Test loading invalid JSON configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_path = Path(f.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                CloudInstrumentConfig.from_file(config_path)
        finally:
            config_path.unlink()
    
    def test_expand_env_vars_nested(self):
        """Test environment variable expansion in nested structures."""
        import os
        
        os.environ["TEST_HOST"] = "example.com"
        os.environ["TEST_PORT"] = "9999"
        
        try:
            data = {
                "osc": {
                    "host": "${TEST_HOST}",
                    "port": 9001  # This should not be expanded
                },
                "database": {
                    "redpanal_url": "http://${TEST_HOST}:${TEST_PORT}/api"
                }
            }
            
            expanded = CloudInstrumentConfig._expand_env_vars(data)
            
            assert expanded["osc"]["host"] == "example.com"
            assert expanded["osc"]["port"] == 9001
            assert expanded["database"]["redpanal_url"] == "http://example.com:9999/api"
            
        finally:
            # Clean up
            for var in ["TEST_HOST", "TEST_PORT"]:
                if var in os.environ:
                    del os.environ[var]
    
    def test_expand_env_vars_missing(self):
        """Test environment variable expansion with missing variables."""
        data = {
            "database": {
                "freesound_api_key": "${MISSING_VAR}"
            }
        }
        
        expanded = CloudInstrumentConfig._expand_env_vars(data)
        # Should return original value if env var not found
        assert expanded["database"]["freesound_api_key"] == "${MISSING_VAR}"