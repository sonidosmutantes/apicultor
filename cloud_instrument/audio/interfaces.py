"""Audio system interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path


class AudioServer(ABC):
    """Abstract base class for audio servers."""
    
    @abstractmethod
    def start(self) -> None:
        """Start the audio server."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the audio server."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the audio server is running."""
        pass
    
    @abstractmethod
    def load_sound(self, sound_path: Path) -> str:
        """Load a sound file.
        
        Args:
            sound_path: Path to sound file
            
        Returns:
            Sound ID for referencing the loaded sound
        """
        pass
    
    @abstractmethod
    def play_sound(self, sound_id: str, **kwargs: Any) -> None:
        """Play a loaded sound.
        
        Args:
            sound_id: ID of the sound to play
            kwargs: Additional playback parameters
        """
        pass
    
    @abstractmethod
    def stop_sound(self, sound_id: Optional[str] = None) -> None:
        """Stop playing sound(s).
        
        Args:
            sound_id: Optional specific sound to stop, None stops all
        """
        pass
    
    @abstractmethod
    def set_parameter(self, param_name: str, value: Any) -> None:
        """Set an audio parameter.
        
        Args:
            param_name: Name of the parameter
            value: Parameter value
        """
        pass
    
    @abstractmethod
    def get_parameter(self, param_name: str) -> Any:
        """Get an audio parameter value.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Parameter value
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get server status information.
        
        Returns:
            Status dictionary
        """
        pass


class AudioEffect(ABC):
    """Abstract base class for audio effects."""
    
    @abstractmethod
    def apply(self, **parameters: Any) -> None:
        """Apply the effect with given parameters.
        
        Args:
            parameters: Effect parameters
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current effect parameters.
        
        Returns:
            Dictionary of current parameters
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset effect to default state."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Effect name."""
        pass