"""Cloud Instrument exceptions."""

from typing import Optional


class CloudInstrumentError(Exception):
    """Base exception for Cloud Instrument errors."""
    
    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.component = component
    
    def __str__(self) -> str:
        if self.component:
            return f"[{self.component}] {self.message}"
        return self.message


class ConfigurationError(CloudInstrumentError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "Configuration")
        self.config_key = config_key


class OSCError(CloudInstrumentError):
    """OSC-related errors."""
    
    def __init__(self, message: str, osc_path: Optional[str] = None):
        super().__init__(message, "OSC")
        self.osc_path = osc_path


class AudioError(CloudInstrumentError):
    """Audio system errors."""
    
    def __init__(self, message: str, backend: Optional[str] = None):
        super().__init__(message, "Audio")
        self.backend = backend


class MIDIError(CloudInstrumentError):
    """MIDI-related errors."""
    
    def __init__(self, message: str, device: Optional[str] = None):
        super().__init__(message, "MIDI")
        self.device = device


class DatabaseError(CloudInstrumentError):
    """Database-related errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message, "Database")
        self.provider = provider


class PluginError(CloudInstrumentError):
    """Plugin system errors."""
    
    def __init__(self, message: str, plugin_name: Optional[str] = None):
        super().__init__(message, "Plugin")
        self.plugin_name = plugin_name