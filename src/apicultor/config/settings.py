"""Main settings module for Apicultor configuration."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .base import (
    APIConfig, AudioConfig, OSCConfig, MIRConfig, 
    LoggingConfig, DatabaseConfig, PluginConfig
)


class Settings:
    """Main settings class that aggregates all configuration."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize settings from file and environment variables.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.api = APIConfig()
        self.audio = AudioConfig()
        self.osc = OSCConfig()
        self.mir = MIRConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        self.plugins = PluginConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        else:
            # Try to load from default locations
            self._load_from_default_locations()
            
        # Override with environment variables
        self.load_from_env()
        
        # Validate all configurations
        self.validate()
    
    def _load_from_default_locations(self) -> None:
        """Load configuration from default locations."""
        default_paths = [
            Path.cwd() / "apicultor.json",
            Path.cwd() / ".apicultor.json",
            Path.home() / ".apicultor" / "config.json",
            Path("/etc/apicultor/config.json"),
        ]
        
        for path in default_paths:
            if path.exists():
                try:
                    self.load_from_file(path)
                    break
                except Exception:
                    continue
    
    def load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            # Load API configuration
            if "api" in config_data:
                api_config = config_data["api"]
                for key, value in api_config.items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
            
            # Load audio configuration
            if "audio" in config_data:
                audio_config = config_data["audio"]
                for key, value in audio_config.items():
                    if hasattr(self.audio, key):
                        setattr(self.audio, key, value)
            
            # Load OSC configuration
            if "osc" in config_data:
                osc_config = config_data["osc"]
                for key, value in osc_config.items():
                    if hasattr(self.osc, key):
                        setattr(self.osc, key, value)
            
            # Load MIR configuration
            if "mir" in config_data:
                mir_config = config_data["mir"]
                for key, value in mir_config.items():
                    if hasattr(self.mir, key):
                        setattr(self.mir, key, value)
            
            # Load logging configuration
            if "logging" in config_data:
                logging_config = config_data["logging"]
                for key, value in logging_config.items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
            
            # Load database configuration
            if "database" in config_data:
                database_config = config_data["database"]
                for key, value in database_config.items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            # Load plugin configuration
            if "plugins" in config_data:
                plugin_config = config_data["plugins"]
                for key, value in plugin_config.items():
                    if hasattr(self.plugins, key):
                        setattr(self.plugins, key, value)
                        
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_file}: {e}")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # API configuration
        if os.getenv("APICULTOR_API_DEFAULT"):
            self.api.default = os.getenv("APICULTOR_API_DEFAULT")
        if os.getenv("APICULTOR_API_TIMEOUT"):
            self.api.timeout = int(os.getenv("APICULTOR_API_TIMEOUT"))
        if os.getenv("APICULTOR_FREESOUND_API_KEY"):
            self.api.freesound_api_key = os.getenv("APICULTOR_FREESOUND_API_KEY")
        if os.getenv("APICULTOR_FREESOUND_BASE_URL"):
            self.api.freesound_base_url = os.getenv("APICULTOR_FREESOUND_BASE_URL")
        if os.getenv("APICULTOR_REDPANAL_URL"):
            self.api.redpanal_url = os.getenv("APICULTOR_REDPANAL_URL")
        
        # Audio configuration
        if os.getenv("APICULTOR_AUDIO_ENGINE"):
            self.audio.engine = os.getenv("APICULTOR_AUDIO_ENGINE")
        if os.getenv("APICULTOR_SAMPLE_RATE"):
            self.audio.sample_rate = int(os.getenv("APICULTOR_SAMPLE_RATE"))
        if os.getenv("APICULTOR_BUFFER_SIZE"):
            self.audio.buffer_size = int(os.getenv("APICULTOR_BUFFER_SIZE"))
        if os.getenv("APICULTOR_CHANNELS"):
            self.audio.channels = int(os.getenv("APICULTOR_CHANNELS"))
        
        # OSC configuration
        if os.getenv("APICULTOR_OSC_PORT"):
            self.osc.port = int(os.getenv("APICULTOR_OSC_PORT"))
        if os.getenv("APICULTOR_OSC_HOST"):
            self.osc.host = os.getenv("APICULTOR_OSC_HOST")
        if os.getenv("APICULTOR_OSC_ENABLED"):
            self.osc.enabled = os.getenv("APICULTOR_OSC_ENABLED").lower() == "true"
        
        # Logging configuration
        if os.getenv("APICULTOR_LOG_LEVEL"):
            self.logging.level = os.getenv("APICULTOR_LOG_LEVEL")
        if os.getenv("APICULTOR_LOG_FORMAT"):
            self.logging.format = os.getenv("APICULTOR_LOG_FORMAT")
        if os.getenv("APICULTOR_LOG_FILE"):
            self.logging.file_path = Path(os.getenv("APICULTOR_LOG_FILE"))
        
        # Database configuration
        if os.getenv("APICULTOR_DATABASE_URL"):
            self.database.url = os.getenv("APICULTOR_DATABASE_URL")
        if os.getenv("APICULTOR_DATABASE_TIMEOUT"):
            self.database.connection_timeout = int(os.getenv("APICULTOR_DATABASE_TIMEOUT"))
        if os.getenv("APICULTOR_DATABASE_MAX_CONNECTIONS"):
            self.database.max_connections = int(os.getenv("APICULTOR_DATABASE_MAX_CONNECTIONS"))
        
        # Plugin configuration
        if os.getenv("APICULTOR_ENABLED_MODULES"):
            enabled_modules = os.getenv("APICULTOR_ENABLED_MODULES").split(",")
            self.plugins.enabled_modules = [m.strip() for m in enabled_modules]
        if os.getenv("APICULTOR_DISABLED_MODULES"):
            disabled_modules = os.getenv("APICULTOR_DISABLED_MODULES").split(",") 
            self.plugins.disabled_modules = [m.strip() for m in disabled_modules]
        if os.getenv("APICULTOR_AUTO_DISCOVER"):
            self.plugins.auto_discover = os.getenv("APICULTOR_AUTO_DISCOVER").lower() == "true"
        if os.getenv("APICULTOR_FAIL_ON_PLUGIN_ERROR"):
            self.plugins.fail_on_plugin_error = os.getenv("APICULTOR_FAIL_ON_PLUGIN_ERROR").lower() == "true"
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.api.validate()
        self.audio.validate()
        self.osc.validate()
        self.mir.validate()
        self.logging.validate()
        self.database.validate()
        self.plugins.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "api": {
                "default": self.api.default,
                "timeout": self.api.timeout,
                "freesound_api_key": self.api.freesound_api_key,
                "freesound_base_url": self.api.freesound_base_url,
                "redpanal_url": self.api.redpanal_url,
            },
            "audio": {
                "engine": self.audio.engine,
                "sample_rate": self.audio.sample_rate,
                "buffer_size": self.audio.buffer_size,
                "channels": self.audio.channels,
            },
            "osc": {
                "port": self.osc.port,
                "host": self.osc.host,
                "enabled": self.osc.enabled,
            },
            "mir": {
                "descriptors": self.mir.descriptors,
                "audio_formats": self.mir.audio_formats,
                "cache_enabled": self.mir.cache_enabled,
                "cache_dir": str(self.mir.cache_dir) if self.mir.cache_dir else None,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": str(self.logging.file_path) if self.logging.file_path else None,
                "max_file_size": self.logging.max_file_size,
                "backup_count": self.logging.backup_count,
            },
            "database": {
                "url": self.database.url,
                "connection_timeout": self.database.connection_timeout,
                "max_connections": self.database.max_connections,
                "cache_ttl": self.database.cache_ttl,
            },
            "plugins": {
                "enabled_modules": self.plugins.enabled_modules,
                "disabled_modules": self.plugins.disabled_modules,
                "auto_discover": self.plugins.auto_discover,
                "fail_on_plugin_error": self.plugins.fail_on_plugin_error,
                "plugin_configs": self.plugins.plugin_configs,
            },
        }
    
    def save_to_file(self, config_file: Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_data = self.to_dict()
        
        # Don't save sensitive data like API keys
        if "api" in config_data and "freesound_api_key" in config_data["api"]:
            config_data["api"]["freesound_api_key"] = "${APICULTOR_FREESOUND_API_KEY}"
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)


# Global settings instance - will be initialized when imported
settings = None

def get_settings() -> Settings:
    """Get or create global settings instance.
    
    Returns:
        Global settings instance
    """
    global settings
    if settings is None:
        settings = Settings()
    return settings