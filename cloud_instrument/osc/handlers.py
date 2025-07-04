"""OSC message handlers and routing."""

import logging
from typing import Dict, Callable, List, Optional, Any
from dataclasses import dataclass
from osc.messages import OSCMessage, OSCMessageType
from core.events import EventManager, EventType

logger = logging.getLogger(__name__)


@dataclass
class OSCHandler:
    """OSC message handler."""
    path: str
    callback: Callable[[OSCMessage], None]
    description: str = ""
    
    def matches(self, message: OSCMessage) -> bool:
        """Check if this handler matches the message path."""
        return self.path == message.path or (
            self.path.endswith("/*") and message.path.startswith(self.path[:-2])
        )


class OSCHandlerRegistry:
    """Registry for OSC message handlers."""
    
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self._handlers: List[OSCHandler] = []
        self._setup_default_handlers()
    
    def register_handler(
        self, 
        path: str, 
        callback: Callable[[OSCMessage], None],
        description: str = ""
    ) -> None:
        """Register an OSC message handler.
        
        Args:
            path: OSC path to handle (supports wildcards with /*)
            callback: Function to call when message received
            description: Optional description of handler
        """
        handler = OSCHandler(path, callback, description)
        self._handlers.append(handler)
        logger.info(f"Registered OSC handler for {path}: {description}")
    
    def unregister_handler(self, path: str, callback: Callable[[OSCMessage], None]) -> None:
        """Unregister an OSC message handler.
        
        Args:
            path: OSC path
            callback: Callback function to remove
        """
        self._handlers = [
            h for h in self._handlers 
            if not (h.path == path and h.callback == callback)
        ]
        logger.debug(f"Unregistered OSC handler for {path}")
    
    def handle_message(self, message: OSCMessage) -> bool:
        """Handle an OSC message by routing to registered handlers.
        
        Args:
            message: OSC message to handle
            
        Returns:
            True if message was handled by at least one handler
        """
        handled = False
        
        # Emit OSC message event
        self.event_manager.emit(
            EventType.OSC_MESSAGE,
            {"message": message.to_dict()},
            source="OSC"
        )
        
        # Find and execute matching handlers
        for handler in self._handlers:
            if handler.matches(message):
                try:
                    handler.callback(message)
                    handled = True
                    logger.info(f"Handled OSC message {message.path} with {handler.description}")
                except Exception as e:
                    logger.error(f"Error in OSC handler for {message.path}: {e}")
                    self.event_manager.emit(
                        EventType.ERROR_OCCURRED,
                        {"error": str(e), "component": "OSC", "path": message.path},
                        source="OSC"
                    )
        
        if not handled:
            logger.warning(f"No handler found for OSC message: {message.path}")
        
        return handled
    
    def get_handlers(self) -> List[OSCHandler]:
        """Get list of all registered handlers."""
        return self._handlers.copy()
    
    def get_handler_info(self) -> List[Dict[str, str]]:
        """Get information about all registered handlers."""
        return [
            {
                "path": h.path,
                "description": h.description or "No description"
            }
            for h in self._handlers
        ]
    
    def _setup_default_handlers(self) -> None:
        """Set up default OSC message handlers."""
        
        # FX handlers
        self.register_handler(
            "/fx/volume",
            self._handle_fx_volume,
            "Set audio volume (0.0-1.0)"
        )
        
        self.register_handler(
            "/fx/pan",
            self._handle_fx_pan,
            "Set audio pan (-1.0 to 1.0)"
        )
        
        self.register_handler(
            "/fx/reverb",
            self._handle_fx_reverb,
            "Set reverb (room_size, damping)"
        )
        
        # MIR handlers
        self.register_handler(
            "/mir/tempo",
            self._handle_mir_tempo,
            "Set target tempo (BPM)"
        )
        
        self.register_handler(
            "/mir/centroid",
            self._handle_mir_centroid,
            "Set spectral centroid target"
        )
        
        self.register_handler(
            "/mir/duration",
            self._handle_mir_duration,
            "Set target duration (seconds)"
        )
        
        self.register_handler(
            "/mir/hfc",
            self._handle_mir_hfc,
            "Set high frequency content target"
        )
        
        # Sound control handlers
        self.register_handler(
            "/sound/search",
            self._handle_sound_search,
            "Search for sounds by query"
        )
        
        self.register_handler(
            "/sound/play",
            self._handle_sound_play,
            "Play sound by ID or path"
        )
        
        self.register_handler(
            "/sound/stop",
            self._handle_sound_stop,
            "Stop currently playing sound"
        )
        
        # System handlers
        self.register_handler(
            "/system/status",
            self._handle_system_status,
            "Get system status"
        )
        
        self.register_handler(
            "/system/shutdown",
            self._handle_system_shutdown,
            "Shutdown the system"
        )
    
    def _handle_fx_volume(self, message: OSCMessage) -> None:
        """Handle volume change."""
        volume = message.get_float_arg(0, 0.0)
        volume = max(0.0, min(1.0, volume))  # Clamp to valid range
        
        self.event_manager.emit(
            EventType.AUDIO_STATE_CHANGE,
            {"parameter": "volume", "value": volume},
            source="OSC"
        )
        logger.info(f"Volume set to {volume}")
    
    def _handle_fx_pan(self, message: OSCMessage) -> None:
        """Handle pan change."""
        pan = message.get_float_arg(0, 0.0)
        pan = max(-1.0, min(1.0, pan))  # Clamp to valid range
        
        self.event_manager.emit(
            EventType.AUDIO_STATE_CHANGE,
            {"parameter": "pan", "value": pan},
            source="OSC"
        )
        logger.info(f"Pan set to {pan}")
    
    def _handle_fx_reverb(self, message: OSCMessage) -> None:
        """Handle reverb change."""
        room_size = message.get_float_arg(0, 0.5)
        damping = message.get_float_arg(1, 0.5)
        
        self.event_manager.emit(
            EventType.AUDIO_STATE_CHANGE,
            {"parameter": "reverb", "room_size": room_size, "damping": damping},
            source="OSC"
        )
        logger.info(f"Reverb set to room_size={room_size}, damping={damping}")
    
    def _handle_mir_tempo(self, message: OSCMessage) -> None:
        """Handle tempo target change."""
        tempo = message.get_float_arg(0, 120.0)
        
        self.event_manager.emit(
            EventType.MIR_STATE_UPDATE,
            {"parameter": "tempo", "value": tempo},
            source="OSC"
        )
        logger.info(f"MIR tempo target set to {tempo} BPM")
    
    def _handle_mir_centroid(self, message: OSCMessage) -> None:
        """Handle spectral centroid target change."""
        centroid = message.get_float_arg(0, 1000.0)
        
        self.event_manager.emit(
            EventType.MIR_STATE_UPDATE,
            {"parameter": "spectral_centroid", "value": centroid},
            source="OSC"
        )
        logger.info(f"MIR spectral centroid target set to {centroid}")
    
    def _handle_mir_duration(self, message: OSCMessage) -> None:
        """Handle duration target change."""
        duration = message.get_float_arg(0, 5.0)
        
        self.event_manager.emit(
            EventType.MIR_STATE_UPDATE,
            {"parameter": "duration", "value": duration},
            source="OSC"
        )
        logger.info(f"MIR duration target set to {duration} seconds")
    
    def _handle_mir_hfc(self, message: OSCMessage) -> None:
        """Handle HFC target change."""
        hfc = message.get_float_arg(0, 0.1)
        
        self.event_manager.emit(
            EventType.MIR_STATE_UPDATE,
            {"parameter": "hfc", "value": hfc},
            source="OSC"
        )
        logger.info(f"MIR HFC target set to {hfc}")
    
    def _handle_sound_search(self, message: OSCMessage) -> None:
        """Handle sound search request."""
        query = message.get_string_arg(0, "")
        
        if query:
            self.event_manager.emit(
                EventType.OSC_MESSAGE,
                {"action": "search_sounds", "query": query},
                source="OSC"
            )
            logger.info(f"Sound search requested: '{query}'")
        else:
            logger.warning("Sound search requested but no query provided")
    
    def _handle_sound_play(self, message: OSCMessage) -> None:
        """Handle sound play request."""
        sound_id = message.get_string_arg(0, "")
        
        if sound_id:
            self.event_manager.emit(
                EventType.SOUND_PLAY_START,
                {"sound_id": sound_id},
                source="OSC"
            )
            logger.info(f"Sound play requested: {sound_id}")
        else:
            logger.warning("Sound play requested but no sound ID provided")
    
    def _handle_sound_stop(self, message: OSCMessage) -> None:
        """Handle sound stop request."""
        self.event_manager.emit(
            EventType.SOUND_PLAY_STOP,
            {},
            source="OSC"
        )
        logger.info("Sound stop requested")
    
    def _handle_system_status(self, message: OSCMessage) -> None:
        """Handle system status request."""
        self.event_manager.emit(
            EventType.OSC_MESSAGE,
            {"action": "get_status"},
            source="OSC"
        )
        logger.info("System status requested")
    
    def _handle_system_shutdown(self, message: OSCMessage) -> None:
        """Handle system shutdown request."""
        self.event_manager.emit(
            EventType.SYSTEM_SHUTDOWN,
            {"initiated_by": "OSC"},
            source="OSC"
        )
        logger.info("System shutdown requested via OSC")