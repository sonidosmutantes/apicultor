"""MIDI manager for handling MIDI input and routing."""

import logging
import platform
from typing import Optional, List, Dict, Any, Callable
import threading

from midi.messages import MIDIMessage
from midi.controllers import MIDIController, create_controller
from core.config import MIDIConfig
from core.events import EventManager, EventType
from core.exceptions import MIDIError

logger = logging.getLogger(__name__)

# Try to import rtmidi
try:
    import rtmidi
    HAS_RTMIDI = True
except ImportError:
    HAS_RTMIDI = False


class MIDIManager:
    """Manages MIDI input and controller routing."""
    
    def __init__(self, config: MIDIConfig, event_manager: EventManager):
        """Initialize MIDI manager.
        
        Args:
            config: MIDI configuration
            event_manager: Event manager instance
        """
        self.config = config
        self.event_manager = event_manager
        
        self._midi_in: Optional[Any] = None
        self._midi_api = self._get_midi_api()
        self._running = False
        self._controllers: Dict[str, MIDIController] = {}
        self._message_handlers: List[Callable[[MIDIMessage], None]] = []
        
        if not HAS_RTMIDI:
            logger.warning("rtmidi not available - MIDI functionality disabled")
            if config.enabled:
                raise MIDIError("rtmidi library not available. Install with: pip install python-rtmidi")
        
        logger.info(f"Initialized MIDI manager (enabled: {config.enabled})")
    
    def _get_midi_api(self) -> Optional[int]:
        """Get appropriate MIDI API for platform."""
        if not HAS_RTMIDI:
            return None
        
        if platform.system() == "Windows":
            return rtmidi.API_WINDOWS_MM
        elif platform.system() == "Darwin":  # macOS
            return rtmidi.API_MACOSX_CORE
        elif platform.system() == "Linux":
            return rtmidi.API_LINUX_ALSA
        else:
            return rtmidi.API_UNSPECIFIED
    
    def start(self) -> None:
        """Start MIDI manager."""
        if not self.config.enabled:
            logger.info("MIDI disabled in configuration")
            return
        
        if not HAS_RTMIDI:
            logger.warning("Cannot start MIDI - rtmidi not available")
            return
        
        if self._running:
            logger.warning("MIDI manager already running")
            return
        
        try:
            self._midi_in = rtmidi.MidiIn(self._midi_api)
            self._midi_in.set_callback(self._handle_midi_message)
            
            # Open MIDI port
            self._open_midi_port()
            
            self._running = True
            logger.info("MIDI manager started successfully")
            
            # Emit MIDI started event
            self.event_manager.emit(
                EventType.MIDI_MESSAGE,
                {"action": "midi_started"},
                source="MIDIManager"
            )
            
        except Exception as e:
            error_msg = f"Failed to start MIDI manager: {e}"
            logger.error(error_msg)
            raise MIDIError(error_msg)
    
    def stop(self) -> None:
        """Stop MIDI manager."""
        if not self._running:
            return
        
        try:
            if self._midi_in:
                self._midi_in.close_port()
                self._midi_in = None
            
            self._running = False
            logger.info("MIDI manager stopped")
            
            # Emit MIDI stopped event
            self.event_manager.emit(
                EventType.MIDI_MESSAGE,
                {"action": "midi_stopped"},
                source="MIDIManager"
            )
            
        except Exception as e:
            logger.error(f"Error stopping MIDI manager: {e}")
    
    def _open_midi_port(self) -> None:
        """Open MIDI input port."""
        if not self._midi_in:
            return
        
        available_ports = self.get_available_ports()
        
        if self.config.input_device:
            # Try to find specified device
            for i, port_name in enumerate(available_ports):
                if self.config.input_device.lower() in port_name.lower():
                    self._midi_in.open_port(i)
                    logger.info(f"Opened MIDI port: {port_name}")
                    return
            
            logger.warning(f"MIDI device '{self.config.input_device}' not found")
        
        # Try to open first available port
        if available_ports:
            self._midi_in.open_port(0)
            logger.info(f"Opened default MIDI port: {available_ports[0]}")
        elif self.config.virtual_port:
            # Create virtual port
            self._midi_in.open_virtual_port(self.config.client_name)
            logger.info(f"Created virtual MIDI port: {self.config.client_name}")
        else:
            logger.warning("No MIDI ports available and virtual port disabled")
    
    def _handle_midi_message(self, message_data: tuple, data: Any = None) -> None:
        """Handle incoming MIDI message.
        
        Args:
            message_data: Tuple containing (message_bytes, timestamp)
            data: Additional data (unused)
        """
        try:
            message_bytes, timestamp = message_data
            
            if not message_bytes:
                return
            
            # Create MIDI message object
            midi_message = MIDIMessage.from_raw_midi(message_bytes)
            
            logger.debug(f"Received MIDI: {midi_message}")
            
            # Emit MIDI message event
            self.event_manager.emit(
                EventType.MIDI_MESSAGE,
                {"message": midi_message.to_dict()},
                source="MIDIManager"
            )
            
            # Route to controllers
            for controller in self._controllers.values():
                controller.handle_message(midi_message)
            
            # Route to custom handlers
            for handler in self._message_handlers:
                try:
                    handler(midi_message)
                except Exception as e:
                    logger.error(f"Error in MIDI message handler: {e}")
            
        except Exception as e:
            logger.error(f"Error handling MIDI message: {e}")
    
    def add_controller(self, name: str, controller_type: str) -> MIDIController:
        """Add MIDI controller.
        
        Args:
            name: Controller name
            controller_type: Controller type ("generic", "yaeltex", "midimix")
            
        Returns:
            Created controller instance
        """
        controller = create_controller(controller_type, self.event_manager)
        self._controllers[name] = controller
        
        logger.info(f"Added MIDI controller: {name} ({controller_type})")
        return controller
    
    def remove_controller(self, name: str) -> None:
        """Remove MIDI controller.
        
        Args:
            name: Controller name to remove
        """
        if name in self._controllers:
            del self._controllers[name]
            logger.info(f"Removed MIDI controller: {name}")
    
    def get_controller(self, name: str) -> Optional[MIDIController]:
        """Get MIDI controller by name.
        
        Args:
            name: Controller name
            
        Returns:
            Controller instance or None
        """
        return self._controllers.get(name)
    
    def add_message_handler(self, handler: Callable[[MIDIMessage], None]) -> None:
        """Add custom MIDI message handler.
        
        Args:
            handler: Function to handle MIDI messages
        """
        self._message_handlers.append(handler)
        logger.debug("Added custom MIDI message handler")
    
    def remove_message_handler(self, handler: Callable[[MIDIMessage], None]) -> None:
        """Remove custom MIDI message handler.
        
        Args:
            handler: Handler function to remove
        """
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
            logger.debug("Removed custom MIDI message handler")
    
    def get_available_ports(self) -> List[str]:
        """Get list of available MIDI input ports.
        
        Returns:
            List of port names
        """
        if not HAS_RTMIDI:
            return []
        
        try:
            temp_midi_in = rtmidi.MidiIn(self._midi_api)
            ports = []
            for i in range(temp_midi_in.get_port_count()):
                ports.append(temp_midi_in.get_port_name(i))
            return ports
        except Exception as e:
            logger.error(f"Error getting MIDI ports: {e}")
            return []
    
    def send_message(self, message_bytes: List[int]) -> None:
        """Send MIDI message (if output is supported).
        
        Args:
            message_bytes: MIDI message bytes to send
        """
        # TODO: Implement MIDI output if needed
        logger.debug(f"Would send MIDI message: {message_bytes}")
    
    @property
    def is_running(self) -> bool:
        """Check if MIDI manager is running."""
        return self._running
    
    @property
    def is_enabled(self) -> bool:
        """Check if MIDI is enabled."""
        return self.config.enabled and HAS_RTMIDI
    
    def get_status(self) -> Dict[str, Any]:
        """Get MIDI manager status.
        
        Returns:
            Status dictionary
        """
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "rtmidi_available": HAS_RTMIDI,
            "controllers": list(self._controllers.keys()),
            "available_ports": self.get_available_ports(),
            "current_port": self._get_current_port_name(),
            "virtual_port": self.config.virtual_port,
            "client_name": self.config.client_name
        }
    
    def _get_current_port_name(self) -> Optional[str]:
        """Get name of currently opened port."""
        if not self._midi_in:
            return None
        
        try:
            if hasattr(self._midi_in, 'get_current_port'):
                return self._midi_in.get_current_port()
            else:
                # Fallback - return configured device or virtual port name
                return self.config.input_device or (
                    self.config.client_name if self.config.virtual_port else None
                )
        except Exception:
            return None


class MockMIDIManager:
    """Mock MIDI manager for testing when rtmidi is not available."""
    
    def __init__(self, config: MIDIConfig, event_manager: EventManager):
        """Initialize mock MIDI manager."""
        self.config = config
        self.event_manager = event_manager
        self._running = False
        self._controllers: Dict[str, MIDIController] = {}
        
        logger.warning("Using mock MIDI manager - rtmidi not available")
    
    def start(self) -> None:
        """Mock start."""
        self._running = True
        logger.info("Mock MIDI manager 'started'")
    
    def stop(self) -> None:
        """Mock stop."""
        self._running = False
        logger.info("Mock MIDI manager 'stopped'")
    
    def add_controller(self, name: str, controller_type: str) -> MIDIController:
        """Mock add controller."""
        controller = create_controller(controller_type, self.event_manager)
        self._controllers[name] = controller
        logger.info(f"Mock added MIDI controller: {name} ({controller_type})")
        return controller
    
    def remove_controller(self, name: str) -> None:
        """Mock remove controller."""
        if name in self._controllers:
            del self._controllers[name]
            logger.info(f"Mock removed MIDI controller: {name}")
    
    def get_controller(self, name: str) -> Optional[MIDIController]:
        """Mock get controller."""
        return self._controllers.get(name)
    
    def add_message_handler(self, handler: Callable[[MIDIMessage], None]) -> None:
        """Mock add handler."""
        logger.debug("Mock added MIDI message handler")
    
    def remove_message_handler(self, handler: Callable[[MIDIMessage], None]) -> None:
        """Mock remove handler."""
        logger.debug("Mock removed MIDI message handler")
    
    def get_available_ports(self) -> List[str]:
        """Mock get ports."""
        return ["Mock MIDI Port 1", "Mock MIDI Port 2"]
    
    def send_message(self, message_bytes: List[int]) -> None:
        """Mock send message."""
        logger.debug(f"Mock send MIDI message: {message_bytes}")
    
    @property
    def is_running(self) -> bool:
        """Mock running status."""
        return self._running
    
    @property
    def is_enabled(self) -> bool:
        """Mock enabled status."""
        return self.config.enabled
    
    def get_status(self) -> Dict[str, Any]:
        """Mock status."""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "rtmidi_available": False,
            "controllers": list(self._controllers.keys()),
            "available_ports": self.get_available_ports(),
            "current_port": "Mock Port",
            "virtual_port": self.config.virtual_port,
            "client_name": self.config.client_name,
            "mock": True
        }


def create_midi_manager(config: MIDIConfig, event_manager: EventManager) -> MIDIManager:
    """Create MIDI manager with fallback to mock if dependencies unavailable.
    
    Args:
        config: MIDI configuration
        event_manager: Event manager instance
        
    Returns:
        MIDI manager instance (real or mock)
    """
    if HAS_RTMIDI and config.enabled:
        try:
            return MIDIManager(config, event_manager)
        except MIDIError:
            logger.warning("Failed to create real MIDI manager, using mock")
            return MockMIDIManager(config, event_manager)
    else:
        return MockMIDIManager(config, event_manager)