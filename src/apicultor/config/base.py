"""Base configuration classes for Apicultor."""

from typing import List, Literal, Optional, Dict, Any
from pathlib import Path
import os


class BaseConfig:
    """Base configuration class with environment variable support."""
    
    @classmethod
    def from_env(cls, prefix: str = "APICULTOR_") -> "BaseConfig":
        """Create configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Configuration instance
        """
        config = cls()
        for key, value in os.environ.items():
            if key.startswith(prefix):
                attr_name = key[len(prefix):].lower()
                if hasattr(config, attr_name):
                    setattr(config, attr_name, value)
        return config


class APIConfig(BaseConfig):
    """API configuration settings."""
    
    def __init__(self):
        self.default: Literal["freesound", "redpanal"] = "freesound"
        self.timeout: int = 30
        self.freesound_api_key: Optional[str] = None
        self.freesound_base_url: str = "https://freesound.org/apiv2"
        self.redpanal_url: str = "http://api.redpanal.org.ar"
        
    def validate(self) -> None:
        """Validate API configuration."""
        if self.default == "freesound" and not self.freesound_api_key:
            raise ValueError("Freesound API key is required when using freesound as default API")
        if self.timeout <= 0:
            raise ValueError("API timeout must be positive")


class AudioConfig(BaseConfig):
    """Audio engine configuration settings."""
    
    def __init__(self):
        self.engine: Literal["pyo", "supercollider"] = "supercollider"
        self.sample_rate: int = 44100
        self.buffer_size: int = 512
        self.channels: int = 2
        
    def validate(self) -> None:
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.channels <= 0:
            raise ValueError("Channels must be positive")


class OSCConfig(BaseConfig):
    """OSC server configuration settings."""
    
    def __init__(self):
        self.port: int = 9001
        self.host: str = "0.0.0.0"
        self.enabled: bool = True
        
    def validate(self) -> None:
        """Validate OSC configuration."""
        if not (1 <= self.port <= 65535):
            raise ValueError("OSC port must be between 1 and 65535")


class MIRConfig(BaseConfig):
    """MIR analysis configuration settings."""
    
    def __init__(self):
        self.descriptors: List[str] = [
            "lowlevel.spectral_centroid",
            "lowlevel.spectral_contrast", 
            "lowlevel.dissonance",
            "lowlevel.hfc",
            "rhythm.bpm"
        ]
        self.audio_formats: List[str] = [".wav", ".mp3", ".ogg", ".flac"]
        self.cache_enabled: bool = True
        self.cache_dir: Optional[Path] = None
        
    def validate(self) -> None:
        """Validate MIR configuration."""
        if not self.descriptors:
            raise ValueError("At least one MIR descriptor is required")
        if not self.audio_formats:
            raise ValueError("At least one audio format is required")


class LoggingConfig(BaseConfig):
    """Logging configuration settings."""
    
    def __init__(self):
        self.level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
        self.format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.file_path: Optional[Path] = None
        self.max_file_size: int = 10 * 1024 * 1024  # 10MB
        self.backup_count: int = 5
        
    def validate(self) -> None:
        """Validate logging configuration."""
        if self.max_file_size <= 0:
            raise ValueError("Max file size must be positive")
        if self.backup_count < 0:
            raise ValueError("Backup count must be non-negative")


class DatabaseConfig(BaseConfig):
    """Database configuration settings."""
    
    def __init__(self):
        self.url: Optional[str] = None
        self.connection_timeout: int = 30
        self.max_connections: int = 10
        self.cache_ttl: int = 3600  # 1 hour
        
    def validate(self) -> None:
        """Validate database configuration."""
        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        if self.cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")


class PluginConfig(BaseConfig):
    """Plugin system configuration settings."""
    
    def __init__(self):
        self.enabled_modules: List[str] = [
            "database"
        ]
        self.disabled_modules: List[str] = [
            "gradients",
            "emotion", 
            "machine_learning"
        ]
        self.auto_discover: bool = True
        self.fail_on_plugin_error: bool = False
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
    def validate(self) -> None:
        """Validate plugin configuration."""
        # Check for conflicts between enabled and disabled lists
        enabled_set = set(self.enabled_modules)
        disabled_set = set(self.disabled_modules)
        conflicts = enabled_set & disabled_set
        if conflicts:
            raise ValueError(f"Modules cannot be both enabled and disabled: {conflicts}")
            
        # Validate individual plugin configs
        for plugin_name, plugin_config in self.plugin_configs.items():
            if not isinstance(plugin_config, dict):
                raise ValueError(f"Plugin config for {plugin_name} must be a dictionary")