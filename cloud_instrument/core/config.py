"""Configuration management for Cloud Instrument."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AudioBackend(Enum):
    """Supported audio backends."""
    SUPERCOLLIDER = "supercollider"
    PYO = "pyo"
    MOCK = "mock"


class DatabaseProvider(Enum):
    """Supported database providers."""
    FREESOUND = "freesound"
    REDPANAL = "redpanal"
    LOCAL = "local"


@dataclass
class OSCConfig:
    """OSC server configuration."""
    host: str = "127.0.0.1"
    port: int = 9001
    enable_logging: bool = True


@dataclass
class AudioConfig:
    """Audio system configuration."""
    backend: AudioBackend = AudioBackend.SUPERCOLLIDER
    sample_rate: int = 44100
    buffer_size: int = 512
    channels: int = 2
    supercollider_port: int = 57120
    supercollider_host: str = "127.0.0.1"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    default_provider: DatabaseProvider = DatabaseProvider.FREESOUND
    freesound_api_key: Optional[str] = None
    redpanal_url: str = "https://redpanal.org/api/audio"
    local_data_dir: str = "./data"
    local_samples_dir: str = "./samples"


@dataclass
class MIDIConfig:
    """MIDI configuration."""
    enabled: bool = True
    input_device: Optional[str] = None
    virtual_port: bool = True
    client_name: str = "CloudInstrument"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = "cloud_instrument.log"
    console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class CloudInstrumentConfig:
    """Main configuration class for Cloud Instrument."""
    osc: OSCConfig = field(default_factory=OSCConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    midi: MIDIConfig = field(default_factory=MIDIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Plugin system integration
    plugin_configs: Dict[str, Any] = field(default_factory=dict)
    enabled_modules: list[str] = field(default_factory=lambda: ["database"])
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "CloudInstrumentConfig":
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            CloudInstrumentConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudInstrumentConfig":
        """Create config from dictionary with environment variable substitution.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            CloudInstrumentConfig instance
        """
        # Expand environment variables
        expanded_data = cls._expand_env_vars(data)
        
        config = cls()
        
        # Update OSC config
        if "osc" in expanded_data:
            osc_data = expanded_data["osc"]
            config.osc = OSCConfig(
                host=osc_data.get("host", config.osc.host),
                port=osc_data.get("port", config.osc.port),
                enable_logging=osc_data.get("enable_logging", config.osc.enable_logging)
            )
        
        # Update audio config
        if "audio" in expanded_data:
            audio_data = expanded_data["audio"]
            backend_str = audio_data.get("backend", config.audio.backend.value)
            try:
                backend = AudioBackend(backend_str)
            except ValueError:
                logger.warning(f"Invalid audio backend '{backend_str}', using default")
                backend = config.audio.backend
            
            config.audio = AudioConfig(
                backend=backend,
                sample_rate=audio_data.get("sample_rate", config.audio.sample_rate),
                buffer_size=audio_data.get("buffer_size", config.audio.buffer_size),
                channels=audio_data.get("channels", config.audio.channels),
                supercollider_port=audio_data.get("supercollider_port", config.audio.supercollider_port),
                supercollider_host=audio_data.get("supercollider_host", config.audio.supercollider_host)
            )
        
        # Update database config
        if "database" in expanded_data:
            db_data = expanded_data["database"]
            provider_str = db_data.get("default_provider", config.database.default_provider.value)
            try:
                provider = DatabaseProvider(provider_str)
            except ValueError:
                logger.warning(f"Invalid database provider '{provider_str}', using default")
                provider = config.database.default_provider
            
            config.database = DatabaseConfig(
                default_provider=provider,
                freesound_api_key=db_data.get("freesound_api_key"),
                redpanal_url=db_data.get("redpanal_url", config.database.redpanal_url),
                local_data_dir=db_data.get("local_data_dir", config.database.local_data_dir),
                local_samples_dir=db_data.get("local_samples_dir", config.database.local_samples_dir)
            )
        
        # Update MIDI config
        if "midi" in expanded_data:
            midi_data = expanded_data["midi"]
            config.midi = MIDIConfig(
                enabled=midi_data.get("enabled", config.midi.enabled),
                input_device=midi_data.get("input_device"),
                virtual_port=midi_data.get("virtual_port", config.midi.virtual_port),
                client_name=midi_data.get("client_name", config.midi.client_name)
            )
        
        # Update logging config
        if "logging" in expanded_data:
            log_data = expanded_data["logging"]
            config.logging = LoggingConfig(
                level=log_data.get("level", config.logging.level),
                file=log_data.get("file", config.logging.file),
                console=log_data.get("console", config.logging.console),
                format=log_data.get("format", config.logging.format)
            )
        
        # Update plugin configs
        config.plugin_configs = expanded_data.get("plugin_configs", {})
        config.enabled_modules = expanded_data.get("enabled_modules", config.enabled_modules)
        
        return config
    
    @staticmethod
    def _expand_env_vars(data: Any) -> Any:
        """Recursively expand environment variables in configuration data.
        
        Args:
            data: Configuration data (dict, list, or str)
            
        Returns:
            Data with environment variables expanded
        """
        if isinstance(data, dict):
            return {key: CloudInstrumentConfig._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [CloudInstrumentConfig._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)  # Return original if env var not found
        else:
            return data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "osc": {
                "host": self.osc.host,
                "port": self.osc.port,
                "enable_logging": self.osc.enable_logging
            },
            "audio": {
                "backend": self.audio.backend.value,
                "sample_rate": self.audio.sample_rate,
                "buffer_size": self.audio.buffer_size,
                "channels": self.audio.channels,
                "supercollider_port": self.audio.supercollider_port,
                "supercollider_host": self.audio.supercollider_host
            },
            "database": {
                "default_provider": self.database.default_provider.value,
                "freesound_api_key": self.database.freesound_api_key,
                "redpanal_url": self.database.redpanal_url,
                "local_data_dir": self.database.local_data_dir,
                "local_samples_dir": self.database.local_samples_dir
            },
            "midi": {
                "enabled": self.midi.enabled,
                "input_device": self.midi.input_device,
                "virtual_port": self.midi.virtual_port,
                "client_name": self.midi.client_name
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "console": self.logging.console,
                "format": self.logging.format
            },
            "plugin_configs": self.plugin_configs,
            "enabled_modules": self.enabled_modules
        }
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def setup_logging(self) -> None:
        """Configure logging based on configuration."""
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        # Clear existing handlers
        logger_root = logging.getLogger()
        for handler in logger_root.handlers[:]:
            logger_root.removeHandler(handler)
        
        handlers = []
        
        # Console handler
        if self.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(self.logging.format)
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        # File handler
        if self.logging.file:
            file_handler = logging.FileHandler(self.logging.file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(self.logging.format)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            format=self.logging.format
        )