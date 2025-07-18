"""Modern OSC server implementation."""

import logging
import asyncio
from typing import Optional, Callable, Any
from datetime import datetime
import threading

try:
    from pythonosc import dispatcher, server, udp_client
    from pythonosc.server import osc
    HAS_PYTHON_OSC = True
except ImportError:
    HAS_PYTHON_OSC = False

from osc.messages import OSCMessage
from osc.handlers import OSCHandlerRegistry
from core.config import OSCConfig
from core.events import EventManager, EventType
from core.exceptions import OSCError

logger = logging.getLogger(__name__)


class OSCServer:
    """Modern OSC server using python-osc."""
    
    def __init__(self, config: OSCConfig, event_manager: EventManager):
        """Initialize OSC server.
        
        Args:
            config: OSC configuration
            event_manager: Event manager instance
            
        Raises:
            OSCError: If python-osc is not available
        """
        if not HAS_PYTHON_OSC:
            raise OSCError("python-osc library not available. Install with: pip install python-osc")
        
        self.config = config
        self.event_manager = event_manager
        self.handler_registry = OSCHandlerRegistry(event_manager)
        
        self._server: Optional[server.ThreadingOSCUDPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._dispatcher = dispatcher.Dispatcher()
        self._running = False
        
        self._setup_dispatcher()
    
    def _setup_dispatcher(self) -> None:
        """Set up OSC message dispatcher."""
        # Map all incoming messages to our handler
        self._dispatcher.set_default_handler(self._handle_osc_message)
        
        # Map specific paths for better routing
        known_paths = [
            "/fx/*", "/mir/*", "/sound/*", "/system/*", "/midi/*"
        ]
        
        for path in known_paths:
            self._dispatcher.map(path, self._handle_osc_message)
    
    def _handle_osc_message(self, path: str, *args: Any) -> None:
        """Handle incoming OSC message.
        
        Args:
            path: OSC path
            args: Message arguments
        """
        try:
            # Create OSC message object
            message = OSCMessage(
                path=path,
                args=list(args),
                timestamp=datetime.now(),
                source_address=None  # Could be added from server context
            )
            
            if self.config.enable_logging:
                logger.info(f"Received OSC message: {path} {args}")
            
            # Route to handlers
            self.handler_registry.handle_message(message)
            
        except Exception as e:
            logger.error(f"Error handling OSC message {path}: {e}")
            self.event_manager.emit(
                EventType.ERROR_OCCURRED,
                {"error": str(e), "component": "OSC", "path": path},
                source="OSC"
            )
    
    def start(self) -> None:
        """Start the OSC server.
        
        Raises:
            OSCError: If server fails to start
        """
        if self._running:
            logger.warning("OSC server is already running")
            return
        
        try:
            self._server = server.ThreadingOSCUDPServer(
                (self.config.host, self.config.port),
                self._dispatcher
            )
            
            # Start server in separate thread
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True
            )
            self._server_thread.start()
            
            self._running = True
            
            logger.info(f"OSC server started on {self.config.host}:{self.config.port}")
            
            # Emit server started event
            self.event_manager.emit(
                EventType.OSC_MESSAGE,
                {"action": "server_started", "host": self.config.host, "port": self.config.port},
                source="OSC"
            )
            
        except Exception as e:
            error_msg = f"Failed to start OSC server: {e}"
            logger.error(error_msg)
            raise OSCError(error_msg)
    
    def stop(self) -> None:
        """Stop the OSC server."""
        if not self._running:
            return
        
        try:
            if self._server:
                self._server.shutdown()
                self._server.server_close()
            
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=2.0)
            
            self._running = False
            self._server = None
            self._server_thread = None
            
            logger.info("OSC server stopped")
            
            # Emit server stopped event
            self.event_manager.emit(
                EventType.OSC_MESSAGE,
                {"action": "server_stopped"},
                source="OSC"
            )
            
        except Exception as e:
            logger.error(f"Error stopping OSC server: {e}")
    
    def send_message(self, host: str, port: int, path: str, *args: Any) -> None:
        """Send OSC message to remote host.
        
        Args:
            host: Target host
            port: Target port
            path: OSC path
            args: Message arguments
        """
        try:
            client = udp_client.SimpleUDPClient(host, port)
            client.send_message(path, args)
            
            if self.config.enable_logging:
                logger.debug(f"Sent OSC message to {host}:{port}: {path} {args}")
                
        except Exception as e:
            logger.error(f"Failed to send OSC message to {host}:{port}: {e}")
            raise OSCError(f"Failed to send OSC message: {e}")
    
    def register_handler(
        self, 
        path: str, 
        callback: Callable[[OSCMessage], None],
        description: str = ""
    ) -> None:
        """Register custom OSC message handler.
        
        Args:
            path: OSC path to handle
            callback: Function to call when message received
            description: Optional description
        """
        self.handler_registry.register_handler(path, callback, description)
    
    def unregister_handler(self, path: str, callback: Callable[[OSCMessage], None]) -> None:
        """Unregister OSC message handler.
        
        Args:
            path: OSC path
            callback: Callback function to remove
        """
        self.handler_registry.unregister_handler(path, callback)
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    @property
    def address(self) -> tuple[str, int]:
        """Get server address."""
        return (self.config.host, self.config.port)
    
    def get_handler_info(self) -> list[dict[str, str]]:
        """Get information about registered handlers."""
        return self.handler_registry.get_handler_info()
    
    def get_status(self) -> dict[str, Any]:
        """Get server status information."""
        return {
            "running": self._running,
            "host": self.config.host,
            "port": self.config.port,
            "logging_enabled": self.config.enable_logging,
            "handlers_count": len(self.handler_registry.get_handlers()),
            "server_thread_alive": self._server_thread.is_alive() if self._server_thread else False
        }


class MockOSCServer:
    """Mock OSC server for testing when python-osc is not available."""
    
    def __init__(self, config: OSCConfig, event_manager: EventManager):
        """Initialize mock OSC server."""
        self.config = config
        self.event_manager = event_manager
        self.handler_registry = OSCHandlerRegistry(event_manager)
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        
        logger.warning("Using mock OSC server - python-osc not available")
    
    def start(self) -> None:
        """Mock start with simple UDP listener."""
        self._running = True
        logger.info(f"Mock OSC server 'started' on {self.config.host}:{self.config.port}")
        
        # Start a simple UDP listener for demonstration
        self._server_thread = threading.Thread(target=self._mock_listener, daemon=True)
        self._server_thread.start()
    
    def _mock_listener(self) -> None:
        """Simple mock UDP listener that logs received data."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)  # Non-blocking with timeout
            sock.bind((self.config.host, self.config.port))
            
            while self._running:
                try:
                    data, addr = sock.recvfrom(1024)
                    if self.config.enable_logging:
                        # Simple mock parsing - just log the raw data
                        logger.info(f"Mock OSC received from {addr}: {len(data)} bytes")
                        # Try to extract some readable info
                        try:
                            decoded = data.decode('utf-8', errors='ignore')
                            if decoded:
                                logger.info(f"Mock OSC data (partial): {decoded[:100]}")
                        except:
                            pass
                        
                        # Parse and handle the actual OSC data
                        self._parse_and_handle_osc_data(data, addr)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.debug(f"Mock OSC listener error: {e}")
                        
        except Exception as e:
            logger.error(f"Mock OSC server failed to bind to {self.config.host}:{self.config.port}: {e}")
        finally:
            try:
                sock.close()
            except:
                pass
    
    def _parse_and_handle_osc_data(self, data: bytes, addr) -> None:
        """Simple OSC data parsing for mock server."""
        try:
            # Basic OSC message parsing - very simplified
            decoded = data.decode('utf-8', errors='ignore')
            
            # Extract the path (first null-terminated string)
            path_end = decoded.find('\x00')
            if path_end > 0:
                path = decoded[:path_end]
                
                # Simple argument extraction - look for common patterns
                args = []
                remaining = decoded[path_end:]
                
                # Look for float arguments (very basic parsing)
                if ',f' in remaining:
                    # Float argument present - try to extract it
                    import struct
                    try:
                        # Find float data (after type tag)
                        type_idx = data.find(b',f')
                        if type_idx >= 0:
                            # Skip to float data (aligned to 4 bytes)
                            float_start = type_idx + 4
                            while float_start % 4 != 0:
                                float_start += 1
                            if float_start + 4 <= len(data):
                                float_val = struct.unpack('>f', data[float_start:float_start+4])[0]
                                args.append(float_val)
                    except:
                        pass
                
                # Look for string arguments
                elif ',s' in remaining:
                    # String argument - extract it
                    try:
                        string_start = remaining.find(',s') + 2
                        while string_start < len(remaining) and remaining[string_start] in '\x00 ':
                            string_start += 1
                        string_end = remaining.find('\x00', string_start)
                        if string_end > string_start:
                            string_arg = remaining[string_start:string_end]
                            args.append(string_arg)
                    except:
                        pass
                
                # Create proper OSC message
                message = OSCMessage(
                    path=path,
                    args=args,
                    timestamp=datetime.now(),
                    source_address=f"{addr[0]}:{addr[1]}"
                )
                
                if self.config.enable_logging:
                    logger.info(f"Mock OSC parsed message: {path} {args}")
                
                # Route to handlers
                self.handler_registry.handle_message(message)
                
        except Exception as e:
            logger.debug(f"Mock OSC parsing error: {e}")
            # Fallback to generic message
            mock_message = OSCMessage(
                path="/mock/unparsed",
                args=["error"],
                timestamp=datetime.now(),
                source_address=f"{addr[0]}:{addr[1]}"
            )
            self.handler_registry.handle_message(mock_message)
    
    def stop(self) -> None:
        """Mock stop."""
        self._running = False
        logger.info("Mock OSC server 'stopped'")
        
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2)
    
    def send_message(self, host: str, port: int, path: str, *args: Any) -> None:
        """Mock send message."""
        if self.config.enable_logging:
            logger.info(f"Mock OSC send to {host}:{port}: {path} {args}")
    
    def register_handler(self, path: str, callback: Callable, description: str = "") -> None:
        """Mock register handler."""
        self.handler_registry.register_handler(path, callback, description)
    
    def unregister_handler(self, path: str, callback: Callable) -> None:
        """Mock unregister handler."""
        self.handler_registry.unregister_handler(path, callback)
    
    @property
    def is_running(self) -> bool:
        """Mock running status."""
        return self._running
    
    @property
    def address(self) -> tuple[str, int]:
        """Mock address."""
        return (self.config.host, self.config.port)
    
    def get_handler_info(self) -> list[dict[str, str]]:
        """Mock handler info."""
        return self.handler_registry.get_handler_info()
    
    def get_status(self) -> dict[str, Any]:
        """Mock status."""
        return {
            "running": self._running,
            "host": self.config.host,
            "port": self.config.port,
            "logging_enabled": self.config.enable_logging,
            "handlers_count": len(self.handler_registry.get_handlers()),
            "mock": True
        }


def create_osc_server(config: OSCConfig, event_manager: EventManager) -> OSCServer:
    """Create OSC server with fallback to mock if dependencies unavailable.
    
    Args:
        config: OSC configuration
        event_manager: Event manager instance
        
    Returns:
        OSC server instance (real or mock)
    """
    try:
        return OSCServer(config, event_manager)
    except OSCError:
        logger.warning("python-osc not available, using mock OSC server")
        return MockOSCServer(config, event_manager)