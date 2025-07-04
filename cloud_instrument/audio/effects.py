"""Audio effect implementations."""

import logging
from typing import Any, Dict, Optional
from audio.interfaces import AudioEffect

logger = logging.getLogger(__name__)


class VolumeEffect(AudioEffect):
    """Volume control effect."""
    
    def __init__(self):
        self._volume: float = 1.0
    
    def apply(self, volume: float = 1.0, **kwargs: Any) -> None:
        """Apply volume change.
        
        Args:
            volume: Volume level (0.0-1.0)
        """
        self._volume = max(0.0, min(1.0, volume))
        logger.debug(f"Volume set to {self._volume}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current volume parameters."""
        return {"volume": self._volume}
    
    def reset(self) -> None:
        """Reset volume to default."""
        self._volume = 1.0
    
    @property
    def name(self) -> str:
        """Effect name."""
        return "volume"
    
    @property
    def volume(self) -> float:
        """Current volume level."""
        return self._volume


class PanEffect(AudioEffect):
    """Stereo panning effect."""
    
    def __init__(self):
        self._pan: float = 0.0
    
    def apply(self, pan: float = 0.0, **kwargs: Any) -> None:
        """Apply pan change.
        
        Args:
            pan: Pan position (-1.0 = left, 0.0 = center, 1.0 = right)
        """
        self._pan = max(-1.0, min(1.0, pan))
        logger.debug(f"Pan set to {self._pan}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current pan parameters."""
        return {"pan": self._pan}
    
    def reset(self) -> None:
        """Reset pan to center."""
        self._pan = 0.0
    
    @property
    def name(self) -> str:
        """Effect name."""
        return "pan"
    
    @property
    def pan(self) -> float:
        """Current pan position."""
        return self._pan


class ReverbEffect(AudioEffect):
    """Reverb effect."""
    
    def __init__(self):
        self._room_size: float = 0.5
        self._damping: float = 0.5
        self._wet_level: float = 0.3
        self._dry_level: float = 0.7
    
    def apply(self, 
              room_size: Optional[float] = None,
              damping: Optional[float] = None, 
              wet_level: Optional[float] = None,
              dry_level: Optional[float] = None,
              **kwargs: Any) -> None:
        """Apply reverb parameters.
        
        Args:
            room_size: Size of reverb room (0.0-1.0)
            damping: Damping factor (0.0-1.0)
            wet_level: Wet signal level (0.0-1.0)
            dry_level: Dry signal level (0.0-1.0)
        """
        if room_size is not None:
            self._room_size = max(0.0, min(1.0, room_size))
        if damping is not None:
            self._damping = max(0.0, min(1.0, damping))
        if wet_level is not None:
            self._wet_level = max(0.0, min(1.0, wet_level))
        if dry_level is not None:
            self._dry_level = max(0.0, min(1.0, dry_level))
        
        logger.debug(f"Reverb: room_size={self._room_size}, damping={self._damping}, "
                    f"wet={self._wet_level}, dry={self._dry_level}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current reverb parameters."""
        return {
            "room_size": self._room_size,
            "damping": self._damping,
            "wet_level": self._wet_level,
            "dry_level": self._dry_level
        }
    
    def reset(self) -> None:
        """Reset reverb to default parameters."""
        self._room_size = 0.5
        self._damping = 0.5
        self._wet_level = 0.3
        self._dry_level = 0.7
    
    @property
    def name(self) -> str:
        """Effect name."""
        return "reverb"
    
    @property
    def room_size(self) -> float:
        """Current room size."""
        return self._room_size
    
    @property
    def damping(self) -> float:
        """Current damping."""
        return self._damping
    
    @property
    def wet_level(self) -> float:
        """Current wet level."""
        return self._wet_level
    
    @property
    def dry_level(self) -> float:
        """Current dry level."""
        return self._dry_level


class DelayEffect(AudioEffect):
    """Delay effect."""
    
    def __init__(self):
        self._delay_time: float = 0.25  # seconds
        self._feedback: float = 0.3
        self._wet_level: float = 0.3
    
    def apply(self,
              delay_time: Optional[float] = None,
              feedback: Optional[float] = None,
              wet_level: Optional[float] = None,
              **kwargs: Any) -> None:
        """Apply delay parameters.
        
        Args:
            delay_time: Delay time in seconds
            feedback: Feedback amount (0.0-1.0)
            wet_level: Wet signal level (0.0-1.0)
        """
        if delay_time is not None:
            self._delay_time = max(0.0, min(2.0, delay_time))
        if feedback is not None:
            self._feedback = max(0.0, min(0.95, feedback))  # Prevent infinite feedback
        if wet_level is not None:
            self._wet_level = max(0.0, min(1.0, wet_level))
        
        logger.debug(f"Delay: time={self._delay_time}, feedback={self._feedback}, "
                    f"wet={self._wet_level}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current delay parameters."""
        return {
            "delay_time": self._delay_time,
            "feedback": self._feedback,
            "wet_level": self._wet_level
        }
    
    def reset(self) -> None:
        """Reset delay to default parameters."""
        self._delay_time = 0.25
        self._feedback = 0.3
        self._wet_level = 0.3
    
    @property
    def name(self) -> str:
        """Effect name."""
        return "delay"
    
    @property
    def delay_time(self) -> float:
        """Current delay time."""
        return self._delay_time
    
    @property
    def feedback(self) -> float:
        """Current feedback amount."""
        return self._feedback
    
    @property
    def wet_level(self) -> float:
        """Current wet level."""
        return self._wet_level


class FilterEffect(AudioEffect):
    """Filter effect (low-pass/high-pass/band-pass)."""
    
    def __init__(self):
        self._frequency: float = 1000.0  # Hz
        self._resonance: float = 0.7
        self._filter_type: str = "lowpass"
    
    def apply(self,
              frequency: Optional[float] = None,
              resonance: Optional[float] = None,
              filter_type: Optional[str] = None,
              **kwargs: Any) -> None:
        """Apply filter parameters.
        
        Args:
            frequency: Cutoff frequency in Hz
            resonance: Filter resonance (0.0-1.0)
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        """
        if frequency is not None:
            self._frequency = max(20.0, min(20000.0, frequency))
        if resonance is not None:
            self._resonance = max(0.0, min(1.0, resonance))
        if filter_type is not None and filter_type in ["lowpass", "highpass", "bandpass"]:
            self._filter_type = filter_type
        
        logger.debug(f"Filter: type={self._filter_type}, freq={self._frequency}, "
                    f"resonance={self._resonance}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current filter parameters."""
        return {
            "frequency": self._frequency,
            "resonance": self._resonance,
            "filter_type": self._filter_type
        }
    
    def reset(self) -> None:
        """Reset filter to default parameters."""
        self._frequency = 1000.0
        self._resonance = 0.7
        self._filter_type = "lowpass"
    
    @property
    def name(self) -> str:
        """Effect name."""
        return "filter"
    
    @property
    def frequency(self) -> float:
        """Current cutoff frequency."""
        return self._frequency
    
    @property
    def resonance(self) -> float:
        """Current resonance."""
        return self._resonance
    
    @property
    def filter_type(self) -> str:
        """Current filter type."""
        return self._filter_type