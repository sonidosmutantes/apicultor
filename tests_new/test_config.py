"""Test cases for configuration management."""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open
from apicultor.config import get_settings, APIConfig, AudioConfig, OSCConfig, MIRConfig
from apicultor.config.settings import Settings


class TestAPIConfig:
    """Test API configuration."""
    
    def test_default_values(self):
        """Test default API configuration values."""
        config = APIConfig()
        assert config.default == "freesound"
        assert config.timeout == 30
        assert config.freesound_api_key is None
        assert config.freesound_base_url == "https://freesound.org/apiv2"
        assert config.redpanal_url == "http://api.redpanal.org.ar"
    
    def test_validation_success(self):
        """Test successful API configuration validation."""
        config = APIConfig()
        config.default = "redpanal"
        config.validate()  # Should not raise
    
    def test_validation_failure_no_api_key(self):
        """Test API configuration validation failure without API key."""
        config = APIConfig()
        config.default = "freesound"
        config.freesound_api_key = None
        
        with pytest.raises(ValueError, match="Freesound API key is required"):
            config.validate()
    
    def test_validation_failure_invalid_timeout(self):
        """Test API configuration validation failure with invalid timeout."""
        config = APIConfig()
        config.timeout = 0
        
        with pytest.raises(ValueError, match="API timeout must be positive"):
            config.validate()


class TestAudioConfig:
    """Test audio configuration."""
    
    def test_default_values(self):
        """Test default audio configuration values."""
        config = AudioConfig()
        assert config.engine == "supercollider"
        assert config.sample_rate == 44100
        assert config.buffer_size == 512
        assert config.channels == 2
    
    def test_validation_success(self):
        """Test successful audio configuration validation."""
        config = AudioConfig()
        config.validate()  # Should not raise
    
    def test_validation_failure_invalid_sample_rate(self):
        """Test audio configuration validation failure with invalid sample rate."""
        config = AudioConfig()
        config.sample_rate = 0
        
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            config.validate()
    
    def test_validation_failure_invalid_buffer_size(self):
        """Test audio configuration validation failure with invalid buffer size."""
        config = AudioConfig()
        config.buffer_size = -1
        
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            config.validate()


class TestOSCConfig:
    """Test OSC configuration."""
    
    def test_default_values(self):
        """Test default OSC configuration values."""
        config = OSCConfig()
        assert config.port == 9001
        assert config.host == "0.0.0.0"
        assert config.enabled is True
    
    def test_validation_success(self):
        """Test successful OSC configuration validation."""
        config = OSCConfig()
        config.validate()  # Should not raise
    
    def test_validation_failure_invalid_port(self):
        """Test OSC configuration validation failure with invalid port."""
        config = OSCConfig()
        config.port = 70000
        
        with pytest.raises(ValueError, match="OSC port must be between 1 and 65535"):
            config.validate()


class TestMIRConfig:
    """Test MIR configuration."""
    
    def test_default_values(self):
        """Test default MIR configuration values."""
        config = MIRConfig()
        assert len(config.descriptors) == 5
        assert "lowlevel.spectral_centroid" in config.descriptors
        assert len(config.audio_formats) == 4
        assert ".wav" in config.audio_formats
        assert config.cache_enabled is True
    
    def test_validation_success(self):
        """Test successful MIR configuration validation."""
        config = MIRConfig()
        config.validate()  # Should not raise
    
    def test_validation_failure_empty_descriptors(self):
        """Test MIR configuration validation failure with empty descriptors."""
        config = MIRConfig()
        config.descriptors = []
        
        with pytest.raises(ValueError, match="At least one MIR descriptor is required"):
            config.validate()


class TestSettings:
    """Test main settings class."""
    
    def test_default_initialization(self):
        """Test default settings initialization."""
        settings = Settings()
        assert isinstance(settings.api, APIConfig)
        assert isinstance(settings.audio, AudioConfig)
        assert isinstance(settings.osc, OSCConfig)
        assert isinstance(settings.mir, MIRConfig)
    
    @patch.dict(os.environ, {
        "APICULTOR_API_DEFAULT": "redpanal",
        "APICULTOR_OSC_PORT": "8000",
        "APICULTOR_SAMPLE_RATE": "48000"
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        settings = Settings()
        assert settings.api.default == "redpanal"
        assert settings.osc.port == 8000
        assert settings.audio.sample_rate == 48000
    
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "api": {"default": "freesound", "timeout": 60},
        "audio": {"engine": "pyo", "sample_rate": 48000},
        "osc": {"port": 8000, "host": "127.0.0.1"}
    }))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_file(self, mock_exists, mock_file):
        """Test loading configuration from JSON file."""
        settings = Settings(Path("test_config.json"))
        assert settings.api.default == "freesound"
        assert settings.api.timeout == 60
        assert settings.audio.engine == "pyo"
        assert settings.audio.sample_rate == 48000
        assert settings.osc.port == 8000
        assert settings.osc.host == "127.0.0.1"
    
    def test_to_dict(self):
        """Test exporting configuration as dictionary."""
        settings = Settings()
        config_dict = settings.to_dict()
        
        assert "api" in config_dict
        assert "audio" in config_dict
        assert "osc" in config_dict
        assert "mir" in config_dict
        assert "logging" in config_dict
        assert "database" in config_dict
        
        assert config_dict["api"]["default"] == "freesound"
        assert config_dict["audio"]["engine"] == "supercollider"
        assert config_dict["osc"]["port"] == 9001
    
    @patch("builtins.open", new_callable=mock_open)
    def test_save_to_file(self, mock_file):
        """Test saving configuration to JSON file."""
        settings = Settings()
        settings.api.freesound_api_key = "test_key"
        
        settings.save_to_file(Path("test_config.json"))
        
        mock_file.assert_called_once_with(Path("test_config.json"), 'w')
        written_data = "".join(call.args[0] for call in mock_file().write.call_args_list)
        config_data = json.loads(written_data)
        
        # API key should be replaced with environment variable placeholder
        assert config_data["api"]["freesound_api_key"] == "${APICULTOR_FREESOUND_API_KEY}"
    
    def test_validation_called(self):
        """Test that validation is called during initialization."""
        with patch.object(APIConfig, 'validate') as mock_validate:
            Settings()
            mock_validate.assert_called_once()


class TestIntegration:
    """Integration tests for configuration system."""
    
    @patch.dict(os.environ, {
        "APICULTOR_FREESOUND_API_KEY": "test_key",
        "APICULTOR_API_DEFAULT": "freesound"
    })
    def test_full_configuration_flow(self):
        """Test complete configuration flow with environment variables."""
        settings = Settings()
        
        # Configuration should be loaded from environment
        assert settings.api.freesound_api_key == "test_key"
        assert settings.api.default == "freesound"
        
        # Validation should pass
        settings.validate()  # Should not raise
        
        # Should be able to export to dict
        config_dict = settings.to_dict()
        assert config_dict["api"]["freesound_api_key"] == "test_key"
    
    def test_invalid_configuration_raises_error(self):
        """Test that invalid configuration raises appropriate errors."""
        with patch.dict(os.environ, {
            "APICULTOR_API_DEFAULT": "freesound",
            "APICULTOR_OSC_PORT": "invalid_port"
        }):
            with pytest.raises(ValueError):
                Settings()  # Should raise due to invalid port