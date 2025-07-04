"""OSC message types and data structures."""

from enum import Enum
from typing import Any, List, Union, Optional
from dataclasses import dataclass
from datetime import datetime


class OSCMessageType(Enum):
    """Types of OSC messages the system handles."""
    # Audio effects
    FX_VOLUME = "/fx/volume"
    FX_PAN = "/fx/pan"
    FX_REVERB = "/fx/reverb"
    FX_DELAY = "/fx/delay"
    FX_FILTER = "/fx/filter"
    
    # MIR (Music Information Retrieval) parameters
    MIR_TEMPO = "/mir/tempo"
    MIR_CENTROID = "/mir/centroid"
    MIR_DURATION = "/mir/duration"
    MIR_HFC = "/mir/hfc"
    MIR_BRIGHTNESS = "/mir/brightness"
    MIR_ROLLOFF = "/mir/rolloff"
    MIR_ENERGY = "/mir/energy"
    
    # Sound control
    SOUND_SEARCH = "/sound/search"
    SOUND_PLAY = "/sound/play"
    SOUND_STOP = "/sound/stop"
    SOUND_LOAD = "/sound/load"
    
    # System control
    SYSTEM_STATUS = "/system/status"
    SYSTEM_SHUTDOWN = "/system/shutdown"
    SYSTEM_RESET = "/system/reset"
    
    # MIDI control
    MIDI_NOTE_ON = "/midi/note_on"
    MIDI_NOTE_OFF = "/midi/note_off"
    MIDI_CC = "/midi/cc"


@dataclass
class OSCMessage:
    """OSC message data structure."""
    path: str
    args: List[Union[int, float, str, bool]]
    timestamp: datetime
    source_address: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def message_type(self) -> Optional[OSCMessageType]:
        """Get the message type enum if recognized."""
        try:
            return OSCMessageType(self.path)
        except ValueError:
            return None
    
    @property
    def is_fx_message(self) -> bool:
        """Check if this is an FX message."""
        return self.path.startswith("/fx/")
    
    @property
    def is_mir_message(self) -> bool:
        """Check if this is a MIR message."""
        return self.path.startswith("/mir/")
    
    @property
    def is_sound_message(self) -> bool:
        """Check if this is a sound control message."""
        return self.path.startswith("/sound/")
    
    @property
    def is_system_message(self) -> bool:
        """Check if this is a system control message."""
        return self.path.startswith("/system/")
    
    @property
    def is_midi_message(self) -> bool:
        """Check if this is a MIDI message."""
        return self.path.startswith("/midi/")
    
    def get_float_arg(self, index: int, default: float = 0.0) -> float:
        """Get argument as float with bounds checking."""
        if index < len(self.args):
            try:
                return float(self.args[index])
            except (ValueError, TypeError):
                return default
        return default
    
    def get_int_arg(self, index: int, default: int = 0) -> int:
        """Get argument as int with bounds checking."""
        if index < len(self.args):
            try:
                return int(self.args[index])
            except (ValueError, TypeError):
                return default
        return default
    
    def get_string_arg(self, index: int, default: str = "") -> str:
        """Get argument as string with bounds checking."""
        if index < len(self.args):
            try:
                return str(self.args[index])
            except (ValueError, TypeError):
                return default
        return default
    
    def get_bool_arg(self, index: int, default: bool = False) -> bool:
        """Get argument as boolean with bounds checking."""
        if index < len(self.args):
            arg = self.args[index]
            if isinstance(arg, bool):
                return arg
            elif isinstance(arg, (int, float)):
                return arg != 0
            elif isinstance(arg, str):
                return arg.lower() in ("true", "1", "yes", "on")
            else:
                return default
        return default
    
    def to_dict(self) -> dict:
        """Convert message to dictionary representation."""
        return {
            "path": self.path,
            "args": self.args,
            "timestamp": self.timestamp.isoformat(),
            "source_address": self.source_address,
            "message_type": self.message_type.value if self.message_type else None
        }