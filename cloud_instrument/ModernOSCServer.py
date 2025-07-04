#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern OSC Server for Cloud Instrument.

Supports both python-osc and liblo libraries with a unified interface.
"""

import logging
import threading
from typing import Optional, Any, Callable, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import OSC libraries
osc_lib = None
try:
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.udp_client import SimpleUDPClient
    osc_lib = "python-osc"
except ImportError:
    try:
        import liblo
        from liblo import make_method
        osc_lib = "liblo"
    except ImportError:
        logger.warning("No OSC library available")
        osc_lib = None


class ModernOSCServer:
    """Modern OSC server with unified interface for different OSC libraries."""
    
    def __init__(self, port: int = 9001):
        """Initialize OSC server.
        
        Args:
            port: OSC server port
        """
        self.port = port
        self.debug = True
        self.audio_server: Optional[Any] = None
        self.plugin_manager: Optional[Any] = None
        self.logger = logger
        
        # Server instances
        self._server: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Message handlers
        self._handlers: Dict[str, Callable] = {}
        
        if osc_lib == "python-osc":
            self._setup_python_osc()
        elif osc_lib == "liblo":
            self._setup_liblo()
        else:
            raise RuntimeError("No OSC library available")
    
    def _setup_python_osc(self) -> None:
        """Set up python-osc server."""
        self._dispatcher = Dispatcher()
        
        # Register default handlers
        self._register_handlers()
        
        self._server = ThreadingOSCUDPServer(
            ("127.0.0.1", self.port), 
            self._dispatcher
        )
        
        self.logger.info(f"python-osc server initialized on port {self.port}")
    
    def _setup_liblo(self) -> None:
        """Set up liblo server (fallback)."""
        # This would be implemented if liblo is available
        # For now, we'll focus on python-osc
        raise RuntimeError("liblo setup not implemented in modern version")
    
    def _register_handlers(self) -> None:
        """Register OSC message handlers."""
        # General FX handlers
        self._dispatcher.map("/fx/volume", self._handle_volume)
        self._dispatcher.map("/fx/pan", self._handle_pan)
        self._dispatcher.map("/fx/reverb", self._handle_reverb)
        
        # MIR state handlers
        self._dispatcher.map("/mir/tempo", self._handle_mir_tempo)
        self._dispatcher.map("/mir/centroid", self._handle_mir_centroid)
        self._dispatcher.map("/mir/duration", self._handle_mir_duration)
        self._dispatcher.map("/mir/hfc", self._handle_mir_hfc)
        
        # Sound retrieval
        self._dispatcher.map("/sound/search", self._handle_sound_search)
        self._dispatcher.map("/sound/play", self._handle_sound_play)
        
        # System control
        self._dispatcher.map("/system/status", self._handle_system_status)
        self._dispatcher.map("/system/shutdown", self._handle_system_shutdown)
        
        # Default handler for unmatched messages
        self._dispatcher.map("/*", self._handle_default)
    
    def start(self) -> None:
        """Start the OSC server."""
        if osc_lib != "python-osc":
            raise RuntimeError("Only python-osc is supported in modern version")
        
        if self._running:
            self.logger.warning("OSC server already running")
            return
        
        try:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
            self._running = True
            
            self.logger.info(f"OSC server started on port {self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start OSC server: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the OSC server."""
        if not self._running:
            return
        
        try:
            if self._server:
                self._server.shutdown()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            
            self._running = False
            self.logger.info("OSC server stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping OSC server: {e}")
    
    # OSC Message Handlers
    
    def _handle_volume(self, address: str, *args) -> None:
        """Handle volume control messages."""
        if args and self.audio_server:
            volume = float(args[0])
            if self.debug:
                print(f"Update volume: {volume}")
            
            if hasattr(self.audio_server, 'set_amp'):
                self.audio_server.set_amp(volume)
    
    def _handle_pan(self, address: str, *args) -> None:
        """Handle pan control messages."""
        if args and self.audio_server:
            pan = float(args[0])
            if self.debug:
                print(f"Update pan: {pan}")
            
            if hasattr(self.audio_server, 'set_pan'):
                self.audio_server.set_pan(pan)
    
    def _handle_reverb(self, address: str, *args) -> None:
        """Handle reverb control messages."""
        if len(args) >= 2 and self.audio_server:
            reverb_send = float(args[0])
            reverb_room = float(args[1])
            if self.debug:
                print(f"Update reverb: send={reverb_send}, room={reverb_room}")
            
            if hasattr(self.audio_server, 'set_reverb'):
                self.audio_server.set_reverb(reverb_send, reverb_room)
    
    def _handle_mir_tempo(self, address: str, *args) -> None:
        """Handle MIR tempo messages."""
        if args:
            tempo = float(args[0])
            if self.debug:
                print(f"MIR tempo: {tempo}")
            
            # Update MIR state through plugin manager
            if self.plugin_manager:
                analysis_plugin = self.plugin_manager.get_plugin("analysis")
                if analysis_plugin and hasattr(analysis_plugin, 'set_tempo'):
                    analysis_plugin.set_tempo(tempo)
    
    def _handle_mir_centroid(self, address: str, *args) -> None:
        """Handle MIR spectral centroid messages."""
        if args:
            centroid = float(args[0])
            if self.debug:
                print(f"MIR spectral centroid: {centroid}")
            
            if self.plugin_manager:
                analysis_plugin = self.plugin_manager.get_plugin("analysis")
                if analysis_plugin and hasattr(analysis_plugin, 'set_spectral_centroid'):
                    analysis_plugin.set_spectral_centroid(centroid)
    
    def _handle_mir_duration(self, address: str, *args) -> None:
        """Handle MIR duration messages."""
        if args:
            duration = float(args[0])
            if self.debug:
                print(f"MIR duration: {duration}")
            
            if self.plugin_manager:
                analysis_plugin = self.plugin_manager.get_plugin("analysis")
                if analysis_plugin and hasattr(analysis_plugin, 'set_duration'):
                    analysis_plugin.set_duration(duration)
    
    def _handle_mir_hfc(self, address: str, *args) -> None:
        """Handle MIR high frequency content messages."""
        if args:
            hfc = float(args[0])
            if self.debug:
                print(f"MIR HFC: {hfc}")
            
            if self.plugin_manager:
                analysis_plugin = self.plugin_manager.get_plugin("analysis")
                if analysis_plugin and hasattr(analysis_plugin, 'set_hfc'):
                    analysis_plugin.set_hfc(hfc)
    
    def _handle_sound_search(self, address: str, *args) -> None:
        """Handle sound search messages."""
        if args:
            query = str(args[0])
            if self.debug:
                print(f"Sound search: {query}")
            
            if self.plugin_manager:
                database_plugin = self.plugin_manager.get_plugin("database")
                if database_plugin and hasattr(database_plugin, 'search_sounds'):
                    results = database_plugin.search_sounds(query, limit=10)
                    print(f"Found {len(results) if results else 0} sounds")
    
    def _handle_sound_play(self, address: str, *args) -> None:
        """Handle sound play messages."""
        if args:
            sound_id = str(args[0])
            if self.debug:
                print(f"Play sound: {sound_id}")
            
            if self.audio_server and hasattr(self.audio_server, 'play_sound'):
                self.audio_server.play_sound(sound_id)
    
    def _handle_system_status(self, address: str, *args) -> None:
        """Handle system status requests."""
        if self.debug:
            print("System status request")
        
        status = {
            "osc_server": "running" if self._running else "stopped",
            "audio_server": "available" if self.audio_server else "unavailable",
            "plugin_manager": "available" if self.plugin_manager else "unavailable"
        }
        
        if self.plugin_manager:
            enabled_plugins = self.plugin_manager.list_enabled_plugins()
            status["enabled_plugins"] = enabled_plugins
        
        print(f"Status: {status}")
    
    def _handle_system_shutdown(self, address: str, *args) -> None:
        """Handle system shutdown messages."""
        if self.debug:
            print("System shutdown requested via OSC")
        
        # This could trigger a graceful shutdown of the entire system
        # For now, just log it
        self.logger.info("Shutdown requested via OSC")
    
    def _handle_default(self, address: str, *args) -> None:
        """Handle unmatched OSC messages."""
        if self.debug:
            print(f"Unhandled OSC message: {address} {args}")
    
    def send_message(self, address: str, *args) -> None:
        """Send an OSC message (for testing/feedback)."""
        # This would be used to send messages back to controllers
        # Implementation would depend on having client addresses
        if self.debug:
            print(f"Would send OSC: {address} {args}")


# For backwards compatibility, create an alias
OSCServer = ModernOSCServer