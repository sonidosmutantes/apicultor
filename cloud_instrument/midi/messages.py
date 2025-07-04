"""MIDI message types and data structures."""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


class MIDIMessageType(Enum):
    """Types of MIDI messages."""
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    CONTROL_CHANGE = "control_change"
    PROGRAM_CHANGE = "program_change"
    PITCH_BEND = "pitch_bend"
    AFTERTOUCH = "aftertouch"
    CHANNEL_PRESSURE = "channel_pressure"
    SYSTEM_EXCLUSIVE = "system_exclusive"


@dataclass
class MIDIMessage:
    """MIDI message data structure."""
    message_type: MIDIMessageType
    channel: int
    data: List[int]
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @classmethod
    def from_raw_midi(cls, raw_message: List[int], timestamp: Optional[datetime] = None) -> "MIDIMessage":
        """Create MIDI message from raw MIDI data.
        
        Args:
            raw_message: Raw MIDI message bytes
            timestamp: Optional timestamp
            
        Returns:
            MIDIMessage instance
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if not raw_message:
            raise ValueError("Empty MIDI message")
        
        status_byte = raw_message[0]
        channel = status_byte & 0x0F
        status_type = status_byte & 0xF0
        
        # Determine message type
        if status_type == 0x90:  # Note On
            message_type = MIDIMessageType.NOTE_ON
        elif status_type == 0x80:  # Note Off
            message_type = MIDIMessageType.NOTE_OFF
        elif status_type == 0xB0:  # Control Change
            message_type = MIDIMessageType.CONTROL_CHANGE
        elif status_type == 0xC0:  # Program Change
            message_type = MIDIMessageType.PROGRAM_CHANGE
        elif status_type == 0xE0:  # Pitch Bend
            message_type = MIDIMessageType.PITCH_BEND
        elif status_type == 0xA0:  # Aftertouch
            message_type = MIDIMessageType.AFTERTOUCH
        elif status_type == 0xD0:  # Channel Pressure
            message_type = MIDIMessageType.CHANNEL_PRESSURE
        elif status_byte == 0xF0:  # System Exclusive
            message_type = MIDIMessageType.SYSTEM_EXCLUSIVE
        else:
            # Default to control change for unknown messages
            message_type = MIDIMessageType.CONTROL_CHANGE
        
        return cls(
            message_type=message_type,
            channel=channel,
            data=raw_message[1:],  # Data bytes without status
            timestamp=timestamp
        )
    
    @property
    def note_number(self) -> Optional[int]:
        """Get note number for note messages."""
        if self.message_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
            return self.data[0] if self.data else None
        return None
    
    @property
    def velocity(self) -> Optional[int]:
        """Get velocity for note messages."""
        if self.message_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
            return self.data[1] if len(self.data) > 1 else None
        return None
    
    @property
    def controller_number(self) -> Optional[int]:
        """Get controller number for CC messages."""
        if self.message_type == MIDIMessageType.CONTROL_CHANGE:
            return self.data[0] if self.data else None
        return None
    
    @property
    def controller_value(self) -> Optional[int]:
        """Get controller value for CC messages."""
        if self.message_type == MIDIMessageType.CONTROL_CHANGE:
            return self.data[1] if len(self.data) > 1 else None
        return None
    
    @property
    def program_number(self) -> Optional[int]:
        """Get program number for program change messages."""
        if self.message_type == MIDIMessageType.PROGRAM_CHANGE:
            return self.data[0] if self.data else None
        return None
    
    @property
    def pitch_bend_value(self) -> Optional[int]:
        """Get pitch bend value (14-bit)."""
        if self.message_type == MIDIMessageType.PITCH_BEND:
            if len(self.data) >= 2:
                # Combine LSB and MSB into 14-bit value
                return self.data[0] | (self.data[1] << 7)
        return None
    
    @property
    def normalized_velocity(self) -> Optional[float]:
        """Get velocity normalized to 0.0-1.0."""
        velocity = self.velocity
        if velocity is not None:
            return velocity / 127.0
        return None
    
    @property
    def normalized_controller_value(self) -> Optional[float]:
        """Get controller value normalized to 0.0-1.0."""
        value = self.controller_value
        if value is not None:
            return value / 127.0
        return None
    
    @property
    def normalized_pitch_bend(self) -> Optional[float]:
        """Get pitch bend normalized to -1.0 to 1.0."""
        pitch_bend = self.pitch_bend_value
        if pitch_bend is not None:
            # Center is 8192, range is 0-16383
            return (pitch_bend - 8192) / 8192.0
        return None
    
    def to_dict(self) -> dict:
        """Convert message to dictionary representation."""
        return {
            "message_type": self.message_type.value,
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "note_number": self.note_number,
            "velocity": self.velocity,
            "controller_number": self.controller_number,
            "controller_value": self.controller_value,
            "program_number": self.program_number,
            "pitch_bend_value": self.pitch_bend_value
        }
    
    def __str__(self) -> str:
        """String representation of MIDI message."""
        if self.message_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
            return f"{self.message_type.value} ch{self.channel} note{self.note_number} vel{self.velocity}"
        elif self.message_type == MIDIMessageType.CONTROL_CHANGE:
            return f"{self.message_type.value} ch{self.channel} cc{self.controller_number}={self.controller_value}"
        elif self.message_type == MIDIMessageType.PROGRAM_CHANGE:
            return f"{self.message_type.value} ch{self.channel} prog{self.program_number}"
        elif self.message_type == MIDIMessageType.PITCH_BEND:
            return f"{self.message_type.value} ch{self.channel} bend{self.pitch_bend_value}"
        else:
            return f"{self.message_type.value} ch{self.channel} data{self.data}"