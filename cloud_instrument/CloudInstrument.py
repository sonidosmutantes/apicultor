#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Cloud Instrument for Apicultor.

This program receives OSC messages to define a sound state based on MIR descriptors
like HFC, BPM, duration, spectral centroid and others.

Features:
* Realtime synthesis using Pyo engine or via SuperCollider
* OSC messages to set MIR state received on configurable port (9001 by default)
* Modern plugin-based architecture with type safety
* Configurable API backends: Freesound, RedPanal, custom
* JSON configuration with environment variable support

Example configuration file (.config.json):
{
  "api": "freesound",
  "osc": {
    "port": 9001
  },
  "sound": {
    "synth": "supercollider"
  },
  "plugins": {
    "enabled_modules": ["database", "analysis"],
    "plugin_configs": {
      "database": {
        "default_provider": "freesound",
        "freesound_api_key": "${APICULTOR_FREESOUND_API_KEY}"
      }
    }
  }
}
"""

import sys
import os
import json
import signal
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from time import sleep

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import OSC libraries - try python-osc first, then liblo
osc_lib = None
try:
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.dispatcher import Dispatcher
    osc_lib = "python-osc"
except ImportError:
    try:
        import liblo
        osc_lib = "liblo"
    except ImportError:
        logging.warning("No OSC library available (python-osc or liblo), OSC functionality will be limited")
        liblo = None

try:
    from apicultor.core.plugin_manager import PluginManager, PluginConfig
except ImportError as e:
    logging.error(f"Failed to import required apicultor modules: {e}")
    logging.error("Please ensure apicultor is properly installed")
    sys.exit(1)

# Cloud instrument specific imports (these may not be available in all setups)
try:
    from ModernOSCServer import ModernOSCServer as OSCServer
except ImportError:
    try:
        from OSCServer import OSCServer
    except ImportError:
        logging.warning("OSCServer not available - OSC functionality will be disabled")
        OSCServer = None

try:
    from synth.SuperColliderServer import SupercolliderServer
except ImportError:
    logging.warning("SuperColliderServer not available - SuperCollider synthesis will be disabled")
    SupercolliderServer = None


class ErrorCode(Enum):
    """Error codes for cloud instrument operation."""
    OK = 0
    NO_CONFIG = 1
    BAD_ARGUMENTS = 3
    BAD_CONFIG = 4
    SERVER_ERROR = 5
    NOT_IMPLEMENTED_ERROR = 6
    PLUGIN_ERROR = 7


class CloudInstrument:
    """Modern Cloud Instrument with plugin-based architecture."""
    
    def __init__(self, config_path: str = ".config.json"):
        """Initialize Cloud Instrument.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.plugin_manager: Optional[PluginManager] = None
        self.osc_server: Optional[OSCServer] = None
        self.audio_server: Optional[SupercolliderServer] = None
        self.logger = logging.getLogger(__name__)
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signal_num: int, frame: Any) -> None:
        """Handle interrupt signals gracefully."""
        self.logger.info('Received interrupt signal, shutting down...')
        self.shutdown()
        sys.exit(ErrorCode.OK.value)
    
    def load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise SystemExit(ErrorCode.NO_CONFIG.value)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            raise SystemExit(ErrorCode.BAD_CONFIG.value)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise SystemExit(ErrorCode.NO_CONFIG.value)
    
    def initialize_plugins(self) -> None:
        """Initialize the plugin system."""
        try:
            # Create plugin configuration
            plugin_config = PluginConfig()
            
            # Load plugin settings from config file
            plugin_settings = self.config.get("plugins", {})
            plugin_config.enabled_plugins = plugin_settings.get(
                "enabled_modules", ["database", "analysis"]
            )
            plugin_config.disabled_plugins = plugin_settings.get(
                "disabled_modules", []
            )
            plugin_config.plugin_configs = plugin_settings.get(
                "plugin_configs", {}
            )
            
            # Add plugins directory to plugin paths
            plugins_dir = Path(__file__).parent.parent / "src" / "apicultor" / "plugins"
            plugin_config.plugin_paths = [str(plugins_dir)]
            
            # Initialize plugin manager
            self.plugin_manager = PluginManager(plugin_config)
            self.plugin_manager.initialize()
            
            self.logger.info(f"Initialized {len(self.plugin_manager.list_enabled_plugins())} plugins")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugins: {e}")
            raise SystemExit(ErrorCode.PLUGIN_ERROR.value)
    
    def setup_database_api(self) -> None:
        """Set up database API based on configuration."""
        api_type = self.config.get("api", "redpanal")
        
        try:
            database_plugin = self.plugin_manager.get_plugin("database")
            if not database_plugin:
                self.logger.warning("Database plugin not available")
                return
            
            if api_type == "freesound":
                freesound_config = self.config.get("Freesound.org", [{}])[0]
                api_key = freesound_config.get("API_KEY", "")
                if not api_key:
                    api_key = os.getenv("APICULTOR_FREESOUND_API_KEY", "")
                
                if api_key:
                    # Configure Freesound API through plugin
                    database_plugin.configure_provider("freesound", {"api_key": api_key})
                    self.logger.info("Freesound API configured")
                else:
                    self.logger.warning("No Freesound API key provided")
            
            elif api_type == "redpanal":
                redpanal_config = self.config.get("RedPanal.org", [{}])[0]
                base_url = redpanal_config.get("url", "http://127.0.0.1:5000")
                database_plugin.configure_provider("redpanal", {"base_url": base_url})
                self.logger.info(f"RedPanal API configured with URL: {base_url}")
            
            else:
                self.logger.warning(f"Unknown API type: {api_type}, using default")
                
        except Exception as e:
            self.logger.error(f"Failed to set up database API: {e}")
    
    def setup_osc_server(self) -> None:
        """Set up OSC server for receiving control messages."""
        if OSCServer is None:
            self.logger.warning("OSC server not available - skipping OSC setup")
            return
            
        osc_port = self.config.get("osc", {}).get("port", 9001)
        
        try:
            self.logger.info(f"Starting OSC server on port {osc_port}")
            self.osc_server = OSCServer(osc_port)
            
            # Attach plugin manager to OSC server for MIR state management
            if hasattr(self.osc_server, 'plugin_manager'):
                self.osc_server.plugin_manager = self.plugin_manager
            
            self.osc_server.start()
            self.logger.info("OSC server started successfully")
            
        except Exception as e:
            if osc_lib == "liblo" and liblo and hasattr(liblo, 'ServerError') and isinstance(e, liblo.ServerError):
                self.logger.error(f"liblo OSC server error: {e}")
            else:
                self.logger.error(f"Failed to start OSC server: {e}")
            self.logger.warning("Continuing without OSC server...")
            self.osc_server = None
    
    def setup_audio_synthesis(self) -> None:
        """Set up audio synthesis engine."""
        sound_synth = self.config.get("sound", {}).get("synth", "supercollider")
        
        if sound_synth == "supercollider":
            if SupercolliderServer is None:
                self.logger.warning("SuperCollider server not available - skipping audio synthesis setup")
                return
                
            try:
                self.audio_server = SupercolliderServer()
                sc_ip = self.config.get("supercollider", {}).get("ip", "127.0.0.1")
                sc_port = self.config.get("supercollider", {}).get("port", 57120)
                
                self.audio_server.start(sc_ip, sc_port)
                
                # Connect audio server with plugin system
                if hasattr(self.audio_server, 'plugin_manager'):
                    self.audio_server.plugin_manager = self.plugin_manager
                
                self.logger.info(f"SuperCollider server started at {sc_ip}:{sc_port}")
                
            except Exception as e:
                self.logger.error(f"Failed to start SuperCollider: {e}")
                self.logger.warning("Continuing without audio synthesis...")
                self.audio_server = None
                
        elif sound_synth == "pyo":
            # Pyo support would be added here if needed
            self.logger.warning("Pyo synthesis not implemented in modern version")
            self.audio_server = None
            
        else:
            self.logger.warning(f"Unknown synthesis engine: {sound_synth}, continuing without audio synthesis")
            self.audio_server = None
    
    def connect_components(self) -> None:
        """Connect OSC server, audio server, and plugin system."""
        connections_made = []
        
        if self.osc_server and self.audio_server:
            # Connect OSC server to audio server
            self.osc_server.audio_server = self.audio_server
            connections_made.append("OSC->Audio")
            
        if self.audio_server and self.plugin_manager:
            # Connect audio server to plugin manager
            self.audio_server.plugin_manager = self.plugin_manager
            connections_made.append("Audio->Plugins")
            
        if self.audio_server:
            # Set up logging for audio server
            self.audio_server.logger = self.logger
            connections_made.append("Audio logging")
            
        if self.osc_server and self.plugin_manager:
            # Connect OSC server to plugin manager
            if hasattr(self.osc_server, 'plugin_manager'):
                self.osc_server.plugin_manager = self.plugin_manager
                connections_made.append("OSC->Plugins")
        
        if connections_made:
            self.logger.info(f"Component connections established: {', '.join(connections_made)}")
        else:
            self.logger.warning("No component connections could be established")
    
    def run(self) -> None:
        """Main run loop for the cloud instrument."""
        self.logger.info("Cloud Instrument started - entering main loop")
        
        sound_synth = self.config.get("sound", {}).get("synth", "supercollider")
        
        try:
            if sound_synth == "pyo":
                # Pyo has its own GUI loop
                # pyo_server.gui(locals())  # Would be implemented if Pyo support added
                pass
            else:
                # Main loop for other synthesis engines
                while True:
                    sleep(5)  # Prevent 100% CPU usage on Raspberry Pi
                    # Add any periodic tasks here
                    
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        self.logger.info("Shutting down Cloud Instrument...")
        
        if self.osc_server:
            try:
                self.osc_server.stop()
                self.logger.info("OSC server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping OSC server: {e}")
        
        if self.audio_server:
            try:
                self.audio_server.stop()
                self.logger.info("Audio server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping audio server: {e}")
        
        if self.plugin_manager:
            try:
                self.plugin_manager.shutdown()
                self.logger.info("Plugin manager shutdown")
            except Exception as e:
                self.logger.error(f"Error shutting down plugins: {e}")
    
    def start(self) -> None:
        """Start the complete cloud instrument system."""
        self.logger.info("Starting Cloud Instrument...")
        
        # Initialize all components
        self.load_config()
        self.initialize_plugins()
        self.setup_database_api()
        self.setup_osc_server()
        self.setup_audio_synthesis()
        self.connect_components()
        
        # Start main loop
        self.run()


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cloud_instrument.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> None:
    """Main entry point for Cloud Instrument."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("APICultor Cloud Instrument - Modern Version")
    logger.info("=" * 50)
    
    # Parse command line arguments
    config_path = ".config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    try:
        # Create and start cloud instrument
        instrument = CloudInstrument(config_path)
        instrument.start()
        
    except SystemExit as e:
        logger.info(f"Cloud Instrument exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(ErrorCode.SERVER_ERROR.value)


if __name__ == '__main__':
    main()