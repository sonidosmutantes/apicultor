"""Main Cloud Instrument application class."""

import sys
import signal
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from core.config import CloudInstrumentConfig
from core.events import EventManager, EventType
from core.exceptions import CloudInstrumentError
from osc.server import create_osc_server
from audio.manager import AudioManager
from midi.manager import create_midi_manager

# Import plugin system
try:
    from apicultor.core.plugin_manager import PluginManager
    from apicultor.config.settings import get_settings
    HAS_PLUGIN_SYSTEM = True
except ImportError as e:
    HAS_PLUGIN_SYSTEM = False
    PluginManager = None
    get_settings = None

logger = logging.getLogger(__name__)


class CloudInstrumentApp:
    """Main Cloud Instrument application."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Cloud Instrument application.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.config.setup_logging()
        
        logger.info("Initializing Cloud Instrument application")
        
        # Initialize core components
        self.event_manager = EventManager()
        self._setup_signal_handlers()
        
        # Initialize subsystems
        self.osc_server = create_osc_server(self.config.osc, self.event_manager)
        self.audio_manager = AudioManager(self.config.audio, self.event_manager)
        self.midi_manager = create_midi_manager(self.config.midi, self.event_manager)
        
        # Initialize plugin system if available
        self.plugin_manager: Optional[PluginManager] = None
        if HAS_PLUGIN_SYSTEM:
            try:
                self._setup_plugin_system()
            except Exception as e:
                logger.warning(f"Failed to initialize plugin system: {e}")
        else:
            logger.warning("Plugin system not available - running in standalone mode")
        
        # Application state
        self._running = False
        self._shutdown_requested = False
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("Cloud Instrument application initialized successfully")
    
    def _load_config(self, config_path: Optional[Path]) -> CloudInstrumentConfig:
        """Load configuration from file or defaults.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            CloudInstrumentConfig instance
        """
        if config_path and config_path.exists():
            try:
                config = CloudInstrumentConfig.from_file(config_path)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        # Use default configuration
        config = CloudInstrumentConfig()
        
        # Try to load from standard locations
        standard_paths = [
            Path(".") / "cloud_instrument_config.json",
            Path(".") / "config.json",
            Path("~").expanduser() / ".cloud_instrument.json"
        ]
        
        for path in standard_paths:
            if path.exists():
                try:
                    config = CloudInstrumentConfig.from_file(path)
                    logger.info(f"Loaded configuration from {path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load config from {path}: {e}")
        
        return config
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True
            
            # Emit shutdown event
            self.event_manager.emit(
                EventType.SYSTEM_SHUTDOWN,
                {"signal": signum, "initiated_by": "signal"},
                source="Application"
            )
        
        # Handle common termination signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # Handle SIGUSR1 for status dump (Unix only)
        if hasattr(signal, 'SIGUSR1'):
            def status_handler(signum, frame):
                logger.info("Received SIGUSR1, dumping status...")
                self._dump_status()
            
            signal.signal(signal.SIGUSR1, status_handler)
    
    def _setup_plugin_system(self) -> None:
        """Setup plugin system integration."""
        if not HAS_PLUGIN_SYSTEM:
            return
        
        try:
            # Get plugin settings
            plugin_settings = get_settings()
            plugin_settings.enabled_modules = self.config.enabled_modules
            plugin_settings.plugin_configs.update(self.config.plugin_configs)
            
            # Initialize plugin manager
            self.plugin_manager = PluginManager(plugin_settings)
            self.plugin_manager.initialize_plugins()
            
            logger.info(f"Plugin system initialized with modules: {self.config.enabled_modules}")
            
        except Exception as e:
            logger.error(f"Failed to setup plugin system: {e}")
            raise CloudInstrumentError(f"Plugin system initialization failed: {e}", "PluginSystem")
    
    def _setup_event_handlers(self) -> None:
        """Setup application-level event handlers."""
        self.event_manager.subscribe(
            EventType.SYSTEM_SHUTDOWN,
            self._handle_shutdown_event
        )
        
        self.event_manager.subscribe(
            EventType.ERROR_OCCURRED,
            self._handle_error_event
        )
        
        self.event_manager.subscribe(
            EventType.OSC_MESSAGE,
            self._handle_osc_event
        )
        
        self.event_manager.subscribe(
            EventType.MIR_STATE_UPDATE,
            self._handle_mir_state_update
        )
    
    def start(self) -> None:
        """Start the Cloud Instrument application."""
        if self._running:
            logger.warning("Application is already running")
            return
        
        try:
            logger.info("Starting Cloud Instrument application...")
            
            # Start subsystems in order
            self._start_subsystems()
            
            # Setup MIDI controllers
            self._setup_midi_controllers()
            
            self._running = True
            
            logger.info("Cloud Instrument application started successfully")
            logger.info(f"OSC server listening on {self.osc_server.address}")
            logger.info(f"Audio backend: {self.audio_manager.backend.value}")
            logger.info(f"MIDI enabled: {self.midi_manager.is_enabled}")
            
            # Emit application started event
            self.event_manager.emit(
                EventType.OSC_MESSAGE,
                {"action": "application_started"},
                source="Application"
            )
            
        except Exception as e:
            error_msg = f"Failed to start application: {e}"
            logger.error(error_msg)
            raise CloudInstrumentError(error_msg, "Application")
    
    def stop(self) -> None:
        """Stop the Cloud Instrument application."""
        if not self._running:
            return
        
        try:
            logger.info("Stopping Cloud Instrument application...")
            
            # Stop subsystems in reverse order
            self._stop_subsystems()
            
            # Shutdown plugin system
            if self.plugin_manager:
                try:
                    self.plugin_manager.shutdown_plugins()
                    logger.info("Plugin system shutdown complete")
                except Exception as e:
                    logger.error(f"Error shutting down plugin system: {e}")
            
            self._running = False
            
            logger.info("Cloud Instrument application stopped")
            
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
    
    def run(self) -> None:
        """Run the Cloud Instrument application (blocking)."""
        try:
            self.start()
            
            # Main loop
            logger.info("Cloud Instrument is running. Press Ctrl+C to stop.")
            
            while self._running and not self._shutdown_requested:
                try:
                    # Sleep briefly to allow signal handling
                    import time
                    time.sleep(0.1)
                    
                    # Check system health
                    if not self._check_system_health():
                        logger.warning("System health check failed")
                        break
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received")
                    self._shutdown_requested = True
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    self._shutdown_requested = True
                    break
            
        except Exception as e:
            logger.error(f"Fatal error in application: {e}")
            raise
        finally:
            self.stop()
    
    def _start_subsystems(self) -> None:
        """Start all subsystems."""
        # Start audio system
        try:
            self.audio_manager.start()
            logger.debug("Audio system started")
        except Exception as e:
            logger.warning(f"Audio system failed to start: {e}")
        
        # Start OSC server
        try:
            self.osc_server.start()
            logger.debug("OSC server started")
        except Exception as e:
            logger.error(f"OSC server failed to start: {e}")
            raise
        
        # Start MIDI system
        try:
            self.midi_manager.start()
            logger.debug("MIDI system started")
        except Exception as e:
            logger.warning(f"MIDI system failed to start: {e}")
    
    def _stop_subsystems(self) -> None:
        """Stop all subsystems."""
        # Stop MIDI system
        try:
            self.midi_manager.stop()
            logger.debug("MIDI system stopped")
        except Exception as e:
            logger.error(f"Error stopping MIDI system: {e}")
        
        # Stop OSC server
        try:
            self.osc_server.stop()
            logger.debug("OSC server stopped")
        except Exception as e:
            logger.error(f"Error stopping OSC server: {e}")
        
        # Stop audio system
        try:
            self.audio_manager.stop()
            logger.debug("Audio system stopped")
        except Exception as e:
            logger.error(f"Error stopping audio system: {e}")
    
    def _setup_midi_controllers(self) -> None:
        """Setup MIDI controllers based on configuration."""
        if not self.midi_manager.is_enabled:
            return
        
        # Add default controllers based on configuration
        available_ports = self.midi_manager.get_available_ports()
        
        # Look for known controllers
        for port in available_ports:
            port_lower = port.lower()
            
            if "yaeltex" in port_lower:
                self.midi_manager.add_controller("yaeltex", "yaeltex")
                logger.info("Added Yaeltex MIDI controller")
            elif "midimix" in port_lower or "akai" in port_lower:
                self.midi_manager.add_controller("midimix", "midimix")
                logger.info("Added Akai Midimix MIDI controller")
        
        # Add generic controller if no specific ones found
        if not self.midi_manager._controllers and available_ports:
            self.midi_manager.add_controller("default", "generic")
            logger.info("Added generic MIDI controller")
    
    def _check_system_health(self) -> bool:
        """Check system health.
        
        Returns:
            True if system is healthy
        """
        try:
            # Check OSC server
            if not self.osc_server.is_running:
                logger.error("OSC server is not running")
                return False
            
            # Check audio system
            if not self.audio_manager.is_running:
                logger.warning("Audio system is not running")
                # Don't fail on audio issues, as it might be expected
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return False
    
    def _dump_status(self) -> None:
        """Dump current status to log."""
        logger.info("=== Cloud Instrument Status ===")
        logger.info(f"Running: {self._running}")
        logger.info(f"Shutdown requested: {self._shutdown_requested}")
        
        # OSC status
        osc_status = self.osc_server.get_status()
        logger.info(f"OSC Server: {osc_status}")
        
        # Audio status
        audio_status = self.audio_manager.get_status()
        logger.info(f"Audio System: {audio_status}")
        
        # MIDI status
        midi_status = self.midi_manager.get_status()
        logger.info(f"MIDI System: {midi_status}")
        
        # Plugin status
        if self.plugin_manager:
            try:
                plugin_status = self.plugin_manager.get_plugin_status()
                logger.info(f"Plugins: {plugin_status}")
            except Exception as e:
                logger.info(f"Plugin status error: {e}")
        
        logger.info("=== End Status ===")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive application status.
        
        Returns:
            Status dictionary
        """
        status = {
            "application": {
                "running": self._running,
                "shutdown_requested": self._shutdown_requested,
                "config_loaded": True
            },
            "osc": self.osc_server.get_status(),
            "audio": self.audio_manager.get_status(),
            "midi": self.midi_manager.get_status()
        }
        
        if self.plugin_manager:
            try:
                status["plugins"] = self.plugin_manager.get_plugin_status()
            except Exception as e:
                status["plugins"] = {"error": str(e)}
        else:
            status["plugins"] = {"available": False}
        
        return status
    
    def _handle_shutdown_event(self, event) -> None:
        """Handle shutdown events."""
        logger.info(f"Shutdown event received from {event.source}")
        self._shutdown_requested = True
    
    def _handle_error_event(self, event) -> None:
        """Handle error events."""
        data = event.data
        component = data.get("component", "Unknown")
        error = data.get("error", "Unknown error")
        
        logger.error(f"Error in {component}: {error}")
        
        # Emit error notification
        # Could be extended to send errors to external monitoring
    
    def _handle_osc_event(self, event) -> None:
        """Handle OSC events."""
        data = event.data
        action = data.get("action")
        
        if action == "get_status":
            # Respond with status via OSC
            status = self.get_status()
            logger.info(f"Status requested via OSC: {status}")
        elif action == "search_sounds":
            # Handle sound search request
            query = data.get("query", "")
            self._handle_sound_search(query)
    
    def _handle_mir_state_update(self, event) -> None:
        """Handle MIR state update events."""
        data = event.data
        parameter = data.get("parameter")
        value = data.get("value")
        
        logger.info(f"MIR state update: {parameter} = {value}")
        
        # TODO: Integrate with MIR analysis and sound selection
        # For now, just log the update
    
    def _handle_sound_search(self, query: str) -> None:
        """Handle sound search request.
        
        Args:
            query: Search query string
        """
        if not self.plugin_manager:
            logger.warning("Sound search requested but plugin system not available")
            return
        
        try:
            # TODO: Implement sound search using database plugin
            logger.info(f"Sound search: '{query}'")
            
            # Emit sound search event
            self.event_manager.emit(
                EventType.OSC_MESSAGE,
                {"action": "sound_search_started", "query": query},
                source="Application"
            )
            
        except Exception as e:
            logger.error(f"Error in sound search: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if application is running."""
        return self._running
    
    @property
    def has_plugin_system(self) -> bool:
        """Check if plugin system is available."""
        return self.plugin_manager is not None