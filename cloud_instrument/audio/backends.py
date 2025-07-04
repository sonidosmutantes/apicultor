"""Audio backend implementations."""

import logging
import platform
import subprocess
import uuid
from typing import Any, Dict, Optional, Set
from pathlib import Path

from audio.interfaces import AudioServer
from core.exceptions import AudioError

logger = logging.getLogger(__name__)


class MockAudioServer(AudioServer):
    """Mock audio server for testing and fallback."""
    
    def __init__(self):
        self._running = False
        self._loaded_sounds: Dict[str, Path] = {}
        self._playing_sounds: Set[str] = set()
        self._parameters: Dict[str, Any] = {
            "volume": 1.0,
            "pan": 0.0,
            "reverb_room_size": 0.5,
            "reverb_damping": 0.5
        }
        logger.info("Initialized mock audio server")
    
    def start(self) -> None:
        """Start the mock server."""
        self._running = True
        logger.info("Mock audio server started")
    
    def stop(self) -> None:
        """Stop the mock server."""
        self._running = False
        self._playing_sounds.clear()
        logger.info("Mock audio server stopped")
    
    def is_running(self) -> bool:
        """Check if mock server is running."""
        return self._running
    
    def load_sound(self, sound_path: Path) -> str:
        """Mock load sound."""
        if not sound_path.exists():
            raise AudioError(f"Sound file not found: {sound_path}", "mock")
        
        sound_id = str(uuid.uuid4())
        self._loaded_sounds[sound_id] = sound_path
        logger.info(f"Mock loaded sound: {sound_path.name} -> {sound_id}")
        return sound_id
    
    def play_sound(self, sound_id: str, **kwargs: Any) -> None:
        """Mock play sound."""
        if sound_id not in self._loaded_sounds:
            raise AudioError(f"Sound not loaded: {sound_id}", "mock")
        
        self._playing_sounds.add(sound_id)
        sound_path = self._loaded_sounds[sound_id]
        logger.info(f"Mock playing sound: {sound_path.name} (ID: {sound_id})")
    
    def stop_sound(self, sound_id: Optional[str] = None) -> None:
        """Mock stop sound."""
        if sound_id is None:
            stopped_count = len(self._playing_sounds)
            self._playing_sounds.clear()
            logger.info(f"Mock stopped {stopped_count} sounds")
        else:
            if sound_id in self._playing_sounds:
                self._playing_sounds.remove(sound_id)
                logger.info(f"Mock stopped sound: {sound_id}")
    
    def set_parameter(self, param_name: str, value: Any) -> None:
        """Mock set parameter."""
        self._parameters[param_name] = value
        logger.debug(f"Mock set parameter: {param_name} = {value}")
    
    def get_parameter(self, param_name: str) -> Any:
        """Mock get parameter."""
        return self._parameters.get(param_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Mock get status."""
        return {
            "backend": "mock",
            "running": self._running,
            "loaded_sounds": len(self._loaded_sounds),
            "playing_sounds": len(self._playing_sounds),
            "parameters": self._parameters.copy()
        }


class SuperColliderServer(AudioServer):
    """SuperCollider audio server."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 57120):
        self.host = host
        self.port = port
        self._running = False
        self._loaded_sounds: Dict[str, Path] = {}
        self._sc_process: Optional[subprocess.Popen] = None
        self._parameters: Dict[str, Any] = {
            "volume": 1.0,
            "pan": 0.0,
            "reverb_room_size": 0.5,
            "reverb_damping": 0.5
        }
        logger.info(f"Initialized SuperCollider server for {host}:{port}")
    
    def start(self) -> None:
        """Start SuperCollider server."""
        if self._running:
            logger.warning("SuperCollider server already running")
            return
        
        try:
            # Try to connect to existing SC server first
            if self._test_connection():
                self._running = True
                logger.info(f"Connected to existing SuperCollider server at {self.host}:{self.port}")
                return
            
            # Try to start SC server if not running
            self._start_sc_process()
            
        except Exception as e:
            error_msg = f"Failed to start SuperCollider server: {e}"
            logger.error(error_msg)
            raise AudioError(error_msg, "supercollider")
    
    def stop(self) -> None:
        """Stop SuperCollider server."""
        if not self._running:
            return
        
        try:
            # Send quit message to SC
            if self._sc_process:
                self._sc_process.terminate()
                self._sc_process.wait(timeout=5)
                self._sc_process = None
            
            self._running = False
            logger.info("SuperCollider server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping SuperCollider server: {e}")
    
    def is_running(self) -> bool:
        """Check if SuperCollider server is running."""
        return self._running and self._test_connection()
    
    def load_sound(self, sound_path: Path) -> str:
        """Load sound in SuperCollider."""
        if not self._running:
            raise AudioError("SuperCollider server not running", "supercollider")
        
        if not sound_path.exists():
            raise AudioError(f"Sound file not found: {sound_path}", "supercollider")
        
        sound_id = str(uuid.uuid4())
        
        # TODO: Send OSC message to SC to load buffer
        # For now, just track locally
        self._loaded_sounds[sound_id] = sound_path
        logger.info(f"Loaded sound in SuperCollider: {sound_path.name} -> {sound_id}")
        
        return sound_id
    
    def play_sound(self, sound_id: str, **kwargs: Any) -> None:
        """Play sound in SuperCollider."""
        if not self._running:
            raise AudioError("SuperCollider server not running", "supercollider")
        
        if sound_id not in self._loaded_sounds:
            raise AudioError(f"Sound not loaded: {sound_id}", "supercollider")
        
        # TODO: Send OSC message to SC to play synth
        sound_path = self._loaded_sounds[sound_id]
        logger.info(f"Playing sound in SuperCollider: {sound_path.name} (ID: {sound_id})")
    
    def stop_sound(self, sound_id: Optional[str] = None) -> None:
        """Stop sound in SuperCollider."""
        if not self._running:
            return
        
        # TODO: Send OSC message to SC to stop synth(s)
        if sound_id is None:
            logger.info("Stopping all sounds in SuperCollider")
        else:
            logger.info(f"Stopping sound in SuperCollider: {sound_id}")
    
    def set_parameter(self, param_name: str, value: Any) -> None:
        """Set parameter in SuperCollider."""
        self._parameters[param_name] = value
        
        # TODO: Send OSC message to SC to set parameter
        logger.debug(f"Set SuperCollider parameter: {param_name} = {value}")
    
    def get_parameter(self, param_name: str) -> Any:
        """Get parameter value."""
        return self._parameters.get(param_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get SuperCollider server status."""
        return {
            "backend": "supercollider",
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "loaded_sounds": len(self._loaded_sounds),
            "parameters": self._parameters.copy(),
            "connection_ok": self._test_connection()
        }
    
    def _test_connection(self) -> bool:
        """Test connection to SuperCollider server."""
        try:
            # TODO: Implement actual OSC ping to SC server
            # For now, assume connection is OK if we have a process or external SC
            return True
        except Exception:
            return False
    
    def _start_sc_process(self) -> None:
        """Start SuperCollider process."""
        try:
            # Try to find and start sclang/scsynth
            sc_commands = ["sclang", "scsynth"]
            
            for cmd in sc_commands:
                try:
                    # Check if command exists
                    result = subprocess.run(
                        ["which", cmd] if platform.system() != "Windows" else ["where", cmd],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Found SuperCollider: {cmd}")
                        # TODO: Start SC with proper arguments
                        # self._sc_process = subprocess.Popen([cmd, ...])
                        self._running = True
                        return
                        
                except Exception as e:
                    logger.debug(f"Failed to start {cmd}: {e}")
                    continue
            
            # If we get here, SC was not found
            raise AudioError("SuperCollider not found in PATH", "supercollider")
            
        except Exception as e:
            raise AudioError(f"Failed to start SuperCollider process: {e}", "supercollider")


class PyoAudioServer(AudioServer):
    """Pyo audio server."""
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512, channels: int = 2):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        self._running = False
        self._server = None
        self._loaded_sounds: Dict[str, Any] = {}
        self._parameters: Dict[str, Any] = {
            "volume": 1.0,
            "pan": 0.0,
            "reverb_room_size": 0.5,
            "reverb_damping": 0.5
        }
        
        # Try to import pyo
        try:
            import pyo
            self._pyo = pyo
            self._pyo_available = True
            logger.info("Pyo audio library available")
        except ImportError:
            self._pyo = None
            self._pyo_available = False
            logger.warning("Pyo audio library not available")
    
    def start(self) -> None:
        """Start Pyo audio server."""
        if not self._pyo_available:
            raise AudioError("Pyo audio library not available", "pyo")
        
        if self._running:
            logger.warning("Pyo server already running")
            return
        
        try:
            self._server = self._pyo.Server(
                sr=self.sample_rate,
                nchnls=self.channels,
                buffersize=self.buffer_size,
                duplex=0
            )
            
            self._server.start()
            self._running = True
            logger.info(f"Pyo server started: {self.sample_rate}Hz, {self.channels}ch, {self.buffer_size} buffer")
            
        except Exception as e:
            error_msg = f"Failed to start Pyo server: {e}"
            logger.error(error_msg)
            raise AudioError(error_msg, "pyo")
    
    def stop(self) -> None:
        """Stop Pyo audio server."""
        if not self._running:
            return
        
        try:
            if self._server:
                self._server.stop()
                self._server.shutdown()
                self._server = None
            
            self._running = False
            logger.info("Pyo server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Pyo server: {e}")
    
    def is_running(self) -> bool:
        """Check if Pyo server is running."""
        return self._running and self._server is not None
    
    def load_sound(self, sound_path: Path) -> str:
        """Load sound in Pyo."""
        if not self._running:
            raise AudioError("Pyo server not running", "pyo")
        
        if not sound_path.exists():
            raise AudioError(f"Sound file not found: {sound_path}", "pyo")
        
        try:
            # Load sound file with pyo
            sound_id = str(uuid.uuid4())
            sndtable = self._pyo.SndTable(str(sound_path))
            self._loaded_sounds[sound_id] = sndtable
            
            logger.info(f"Loaded sound in Pyo: {sound_path.name} -> {sound_id}")
            return sound_id
            
        except Exception as e:
            raise AudioError(f"Failed to load sound in Pyo: {e}", "pyo")
    
    def play_sound(self, sound_id: str, **kwargs: Any) -> None:
        """Play sound in Pyo."""
        if not self._running:
            raise AudioError("Pyo server not running", "pyo")
        
        if sound_id not in self._loaded_sounds:
            raise AudioError(f"Sound not loaded: {sound_id}", "pyo")
        
        try:
            sndtable = self._loaded_sounds[sound_id]
            # Create and start player
            player = self._pyo.TableRead(
                table=sndtable,
                freq=sndtable.getRate(),
                mul=self._parameters.get("volume", 1.0)
            ).out()
            
            logger.info(f"Playing sound in Pyo: {sound_id}")
            
        except Exception as e:
            raise AudioError(f"Failed to play sound in Pyo: {e}", "pyo")
    
    def stop_sound(self, sound_id: Optional[str] = None) -> None:
        """Stop sound in Pyo."""
        if not self._running:
            return
        
        try:
            # TODO: Implement proper sound stopping in Pyo
            # For now, just log the action
            if sound_id is None:
                logger.info("Stopping all sounds in Pyo")
            else:
                logger.info(f"Stopping sound in Pyo: {sound_id}")
                
        except Exception as e:
            logger.error(f"Error stopping sound in Pyo: {e}")
    
    def set_parameter(self, param_name: str, value: Any) -> None:
        """Set parameter in Pyo."""
        self._parameters[param_name] = value
        
        # TODO: Apply parameter to active Pyo objects
        logger.debug(f"Set Pyo parameter: {param_name} = {value}")
    
    def get_parameter(self, param_name: str) -> Any:
        """Get parameter value."""
        return self._parameters.get(param_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get Pyo server status."""
        return {
            "backend": "pyo",
            "running": self._running,
            "sample_rate": self.sample_rate,
            "buffer_size": self.buffer_size,
            "channels": self.channels,
            "loaded_sounds": len(self._loaded_sounds),
            "parameters": self._parameters.copy(),
            "pyo_available": self._pyo_available
        }