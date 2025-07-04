"""Audio manager for coordinating audio system components."""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from audio.interfaces import AudioServer, AudioEffect
from audio.backends import SuperColliderServer, PyoAudioServer, MockAudioServer
from audio.effects import VolumeEffect, PanEffect, ReverbEffect, DelayEffect, FilterEffect
from core.config import AudioConfig, AudioBackend
from core.events import EventManager, EventType
from core.exceptions import AudioError

logger = logging.getLogger(__name__)


class AudioManager:
    """Manages audio system components and coordination."""
    
    def __init__(self, config: AudioConfig, event_manager: EventManager):
        """Initialize audio manager.
        
        Args:
            config: Audio configuration
            event_manager: Event manager instance
        """
        self.config = config
        self.event_manager = event_manager
        
        # Initialize audio server
        self._server = self._create_audio_server()
        
        # Initialize effects
        self._effects: Dict[str, AudioEffect] = {
            "volume": VolumeEffect(),
            "pan": PanEffect(),
            "reverb": ReverbEffect(),
            "delay": DelayEffect(),
            "filter": FilterEffect()
        }
        
        # Subscribe to events
        self._setup_event_handlers()
        
        logger.info(f"Initialized audio manager with {config.backend.value} backend")
    
    def _create_audio_server(self) -> AudioServer:
        """Create audio server based on configuration."""
        try:
            if self.config.backend == AudioBackend.SUPERCOLLIDER:
                return SuperColliderServer(
                    host=self.config.supercollider_host,
                    port=self.config.supercollider_port
                )
            elif self.config.backend == AudioBackend.PYO:
                return PyoAudioServer(
                    sample_rate=self.config.sample_rate,
                    buffer_size=self.config.buffer_size,
                    channels=self.config.channels
                )
            else:  # MOCK or fallback
                return MockAudioServer()
                
        except Exception as e:
            logger.warning(f"Failed to create {self.config.backend.value} server: {e}")
            logger.info("Falling back to mock audio server")
            return MockAudioServer()
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for audio system."""
        self.event_manager.subscribe(
            EventType.AUDIO_STATE_CHANGE,
            self._handle_audio_state_change
        )
        
        self.event_manager.subscribe(
            EventType.SOUND_PLAY_START,
            self._handle_sound_play_start
        )
        
        self.event_manager.subscribe(
            EventType.SOUND_PLAY_STOP,
            self._handle_sound_play_stop
        )
        
        self.event_manager.subscribe(
            EventType.SYSTEM_SHUTDOWN,
            self._handle_system_shutdown
        )
    
    def start(self) -> None:
        """Start the audio system."""
        try:
            self._server.start()
            
            # Apply initial effect settings
            self._apply_initial_effects()
            
            logger.info("Audio system started successfully")
            
            # Emit audio system started event
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"action": "system_started", "backend": self.config.backend.value},
                source="AudioManager"
            )
            
        except Exception as e:
            error_msg = f"Failed to start audio system: {e}"
            logger.error(error_msg)
            raise AudioError(error_msg, self.config.backend.value)
    
    def stop(self) -> None:
        """Stop the audio system."""
        try:
            self._server.stop()
            logger.info("Audio system stopped")
            
            # Emit audio system stopped event
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"action": "system_stopped"},
                source="AudioManager"
            )
            
        except Exception as e:
            logger.error(f"Error stopping audio system: {e}")
    
    def load_sound(self, sound_path: Path) -> str:
        """Load a sound file.
        
        Args:
            sound_path: Path to sound file
            
        Returns:
            Sound ID for referencing the loaded sound
            
        Raises:
            AudioError: If loading fails
        """
        try:
            sound_id = self._server.load_sound(sound_path)
            
            # Emit sound loaded event
            self.event_manager.emit(
                EventType.SOUND_LOADED,
                {"sound_id": sound_id, "path": str(sound_path)},
                source="AudioManager"
            )
            
            return sound_id
            
        except Exception as e:
            error_msg = f"Failed to load sound {sound_path}: {e}"
            logger.error(error_msg)
            raise AudioError(error_msg, self.config.backend.value)
    
    def play_sound(self, sound_id: str, **kwargs: Any) -> None:
        """Play a loaded sound.
        
        Args:
            sound_id: ID of the sound to play
            kwargs: Additional playback parameters
        """
        try:
            self._server.play_sound(sound_id, **kwargs)
            
            # Emit sound play event
            self.event_manager.emit(
                EventType.SOUND_PLAY_START,
                {"sound_id": sound_id, "parameters": kwargs},
                source="AudioManager"
            )
            
        except Exception as e:
            error_msg = f"Failed to play sound {sound_id}: {e}"
            logger.error(error_msg)
            raise AudioError(error_msg, self.config.backend.value)
    
    def stop_sound(self, sound_id: Optional[str] = None) -> None:
        """Stop playing sound(s).
        
        Args:
            sound_id: Optional specific sound to stop, None stops all
        """
        try:
            self._server.stop_sound(sound_id)
            
            # Emit sound stop event
            self.event_manager.emit(
                EventType.SOUND_PLAY_STOP,
                {"sound_id": sound_id},
                source="AudioManager"
            )
            
        except Exception as e:
            logger.error(f"Error stopping sound: {e}")
    
    def set_effect_parameter(self, effect_name: str, **parameters: Any) -> None:
        """Set effect parameters.
        
        Args:
            effect_name: Name of the effect
            parameters: Effect parameters to set
        """
        if effect_name not in self._effects:
            raise AudioError(f"Unknown effect: {effect_name}", "AudioManager")
        
        try:
            effect = self._effects[effect_name]
            effect.apply(**parameters)
            
            # Apply to audio server
            self._apply_effect_to_server(effect_name, effect)
            
            logger.debug(f"Applied {effect_name} effect: {parameters}")
            
        except Exception as e:
            error_msg = f"Failed to apply {effect_name} effect: {e}"
            logger.error(error_msg)
            raise AudioError(error_msg, "AudioManager")
    
    def get_effect_parameters(self, effect_name: str) -> Dict[str, Any]:
        """Get current effect parameters.
        
        Args:
            effect_name: Name of the effect
            
        Returns:
            Dictionary of current effect parameters
        """
        if effect_name not in self._effects:
            raise AudioError(f"Unknown effect: {effect_name}", "AudioManager")
        
        return self._effects[effect_name].get_parameters()
    
    def get_available_effects(self) -> List[str]:
        """Get list of available effects.
        
        Returns:
            List of effect names
        """
        return list(self._effects.keys())
    
    def reset_effect(self, effect_name: str) -> None:
        """Reset effect to default parameters.
        
        Args:
            effect_name: Name of the effect to reset
        """
        if effect_name not in self._effects:
            raise AudioError(f"Unknown effect: {effect_name}", "AudioManager")
        
        effect = self._effects[effect_name]
        effect.reset()
        self._apply_effect_to_server(effect_name, effect)
        
        logger.info(f"Reset {effect_name} effect to defaults")
    
    def reset_all_effects(self) -> None:
        """Reset all effects to default parameters."""
        for effect_name in self._effects:
            self.reset_effect(effect_name)
        
        logger.info("Reset all effects to defaults")
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio system status.
        
        Returns:
            Status dictionary
        """
        server_status = self._server.get_status()
        
        # Add effect status
        effects_status = {}
        for name, effect in self._effects.items():
            effects_status[name] = effect.get_parameters()
        
        return {
            "server": server_status,
            "effects": effects_status,
            "config": {
                "backend": self.config.backend.value,
                "sample_rate": self.config.sample_rate,
                "buffer_size": self.config.buffer_size,
                "channels": self.config.channels
            }
        }
    
    @property
    def is_running(self) -> bool:
        """Check if audio system is running."""
        return self._server.is_running()
    
    @property
    def backend(self) -> AudioBackend:
        """Get current audio backend."""
        return self.config.backend
    
    def _apply_initial_effects(self) -> None:
        """Apply initial effect settings."""
        for effect_name, effect in self._effects.items():
            self._apply_effect_to_server(effect_name, effect)
    
    def _apply_effect_to_server(self, effect_name: str, effect: AudioEffect) -> None:
        """Apply effect parameters to audio server.
        
        Args:
            effect_name: Name of the effect
            effect: Effect instance
        """
        parameters = effect.get_parameters()
        
        # Map effect parameters to server parameters
        if effect_name == "volume":
            self._server.set_parameter("volume", parameters["volume"])
        elif effect_name == "pan":
            self._server.set_parameter("pan", parameters["pan"])
        elif effect_name == "reverb":
            self._server.set_parameter("reverb_room_size", parameters["room_size"])
            self._server.set_parameter("reverb_damping", parameters["damping"])
            self._server.set_parameter("reverb_wet", parameters["wet_level"])
            self._server.set_parameter("reverb_dry", parameters["dry_level"])
        elif effect_name == "delay":
            self._server.set_parameter("delay_time", parameters["delay_time"])
            self._server.set_parameter("delay_feedback", parameters["feedback"])
            self._server.set_parameter("delay_wet", parameters["wet_level"])
        elif effect_name == "filter":
            self._server.set_parameter("filter_frequency", parameters["frequency"])
            self._server.set_parameter("filter_resonance", parameters["resonance"])
            self._server.set_parameter("filter_type", parameters["filter_type"])
    
    def _handle_audio_state_change(self, event) -> None:
        """Handle audio state change events."""
        data = event.data
        parameter = data.get("parameter")
        
        if parameter == "volume":
            self.set_effect_parameter("volume", volume=data["value"])
        elif parameter == "pan":
            self.set_effect_parameter("pan", pan=data["value"])
        elif parameter == "reverb":
            self.set_effect_parameter(
                "reverb",
                room_size=data.get("room_size"),
                damping=data.get("damping")
            )
    
    def _handle_sound_play_start(self, event) -> None:
        """Handle sound play start events."""
        data = event.data
        sound_id = data.get("sound_id")
        
        if sound_id:
            try:
                self.play_sound(sound_id)
            except Exception as e:
                logger.error(f"Failed to play sound from event: {e}")
    
    def _handle_sound_play_stop(self, event) -> None:
        """Handle sound play stop events."""
        data = event.data
        sound_id = data.get("sound_id")
        
        try:
            self.stop_sound(sound_id)
        except Exception as e:
            logger.error(f"Failed to stop sound from event: {e}")
    
    def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown events."""
        logger.info("Shutting down audio system due to system shutdown event")
        self.stop()