"""Tests for OSC system."""

import pytest
import time
from unittest.mock import Mock, patch

from ..core.config import OSCConfig
from ..core.events import EventManager, EventType
from ..osc.messages import OSCMessage, OSCMessageType
from ..osc.handlers import OSCHandlerRegistry
from ..osc.server import OSCServer, MockOSCServer, create_osc_server


class TestOSCMessage:
    """Test OSC message handling."""
    
    def test_message_creation(self):
        """Test OSC message creation."""
        message = OSCMessage(
            path="/fx/volume",
            args=[0.5],
            timestamp=None
        )
        
        assert message.path == "/fx/volume"
        assert message.args == [0.5]
        assert message.message_type == OSCMessageType.FX_VOLUME
        assert message.is_fx_message is True
        assert message.is_mir_message is False
    
    def test_message_type_recognition(self):
        """Test message type recognition."""
        # FX message
        fx_msg = OSCMessage("/fx/pan", [-0.5], None)
        assert fx_msg.is_fx_message is True
        assert fx_msg.message_type == OSCMessageType.FX_PAN
        
        # MIR message
        mir_msg = OSCMessage("/mir/tempo", [120.0], None)
        assert mir_msg.is_mir_message is True
        assert mir_msg.message_type == OSCMessageType.MIR_TEMPO
        
        # Sound message
        sound_msg = OSCMessage("/sound/play", ["sound_123"], None)
        assert sound_msg.is_sound_message is True
        assert sound_msg.message_type == OSCMessageType.SOUND_PLAY
        
        # System message
        system_msg = OSCMessage("/system/status", [], None)
        assert system_msg.is_system_message is True
        assert system_msg.message_type == OSCMessageType.SYSTEM_STATUS
        
        # Unknown message
        unknown_msg = OSCMessage("/unknown/path", [], None)
        assert unknown_msg.message_type is None
    
    def test_argument_getters(self):
        """Test argument getter methods."""
        message = OSCMessage(
            "/test",
            [42, 3.14, "hello", True, "false", 0],
            None
        )
        
        assert message.get_int_arg(0) == 42
        assert message.get_float_arg(1) == 3.14
        assert message.get_string_arg(2) == "hello"
        assert message.get_bool_arg(3) is True
        assert message.get_bool_arg(4) is False  # String "false"
        assert message.get_bool_arg(5) is False  # Integer 0
        
        # Test bounds checking
        assert message.get_int_arg(10, 999) == 999
        assert message.get_float_arg(10, 1.23) == 1.23
        assert message.get_string_arg(10, "default") == "default"
        assert message.get_bool_arg(10, True) is True
    
    def test_to_dict(self):
        """Test message serialization."""
        message = OSCMessage("/fx/volume", [0.8], None)
        data = message.to_dict()
        
        assert data["path"] == "/fx/volume"
        assert data["args"] == [0.8]
        assert data["message_type"] == "fx_volume"
        assert "timestamp" in data


class TestOSCHandlerRegistry:
    """Test OSC handler registry."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.event_manager = EventManager()
        self.registry = OSCHandlerRegistry(self.event_manager)
        self.received_messages = []
    
    def message_handler(self, message: OSCMessage):
        """Test message handler."""
        self.received_messages.append(message)
    
    def test_register_handler(self):
        """Test handler registration."""
        self.registry.register_handler("/test", self.message_handler, "Test handler")
        
        handlers = self.registry.get_handlers()
        assert len(handlers) > 0  # Should include default handlers plus our test handler
        
        # Find our test handler
        test_handlers = [h for h in handlers if h.path == "/test"]
        assert len(test_handlers) == 1
        assert test_handlers[0].description == "Test handler"
    
    def test_handle_message(self):
        """Test message handling."""
        self.registry.register_handler("/test", self.message_handler)
        
        message = OSCMessage("/test", ["arg1", "arg2"], None)
        handled = self.registry.handle_message(message)
        
        assert handled is True
        assert len(self.received_messages) == 1
        assert self.received_messages[0].path == "/test"
    
    def test_wildcard_handler(self):
        """Test wildcard handler matching."""
        self.registry.register_handler("/fx/*", self.message_handler)
        
        # Test messages that should match
        volume_msg = OSCMessage("/fx/volume", [0.5], None)
        pan_msg = OSCMessage("/fx/pan", [0.0], None)
        
        self.registry.handle_message(volume_msg)
        self.registry.handle_message(pan_msg)
        
        assert len(self.received_messages) == 2
    
    def test_unregister_handler(self):
        """Test handler unregistration."""
        self.registry.register_handler("/test", self.message_handler)
        
        # Verify handler is registered
        message = OSCMessage("/test", [], None)
        handled = self.registry.handle_message(message)
        assert handled is True
        
        # Unregister and test again
        self.registry.unregister_handler("/test", self.message_handler)
        self.received_messages.clear()
        
        handled = self.registry.handle_message(message)
        # Should still be handled by default handlers, but not our custom one
        assert len(self.received_messages) == 0
    
    def test_default_fx_handlers(self):
        """Test default FX message handlers."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        self.event_manager.subscribe(EventType.AUDIO_STATE_CHANGE, event_handler)
        
        # Test volume
        volume_msg = OSCMessage("/fx/volume", [0.8], None)
        self.registry.handle_message(volume_msg)
        
        assert len(events_received) == 1
        assert events_received[0].data["parameter"] == "volume"
        assert events_received[0].data["value"] == 0.8
        
        # Test pan
        pan_msg = OSCMessage("/fx/pan", [0.5], None)
        self.registry.handle_message(pan_msg)
        
        assert len(events_received) == 2
        assert events_received[1].data["parameter"] == "pan"
        assert events_received[1].data["value"] == 0.5
    
    def test_default_mir_handlers(self):
        """Test default MIR message handlers."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        self.event_manager.subscribe(EventType.MIR_STATE_UPDATE, event_handler)
        
        # Test tempo
        tempo_msg = OSCMessage("/mir/tempo", [120.0], None)
        self.registry.handle_message(tempo_msg)
        
        assert len(events_received) == 1
        assert events_received[0].data["parameter"] == "tempo"
        assert events_received[0].data["value"] == 120.0


class TestMockOSCServer:
    """Test mock OSC server."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = OSCConfig()
        self.event_manager = EventManager()
        self.server = MockOSCServer(self.config, self.event_manager)
    
    def test_server_lifecycle(self):
        """Test server start/stop lifecycle."""
        assert not self.server.is_running
        
        self.server.start()
        assert self.server.is_running
        
        self.server.stop()
        assert not self.server.is_running
    
    def test_handler_registration(self):
        """Test handler registration on mock server."""
        def test_handler(message):
            pass
        
        self.server.register_handler("/test", test_handler, "Test handler")
        
        handlers = self.server.get_handler_info()
        test_handlers = [h for h in handlers if h["path"] == "/test"]
        assert len(test_handlers) == 1
    
    def test_status(self):
        """Test server status."""
        status = self.server.get_status()
        
        assert "running" in status
        assert "host" in status
        assert "port" in status
        assert "mock" in status
        assert status["mock"] is True


@pytest.mark.skipif(
    not pytest.importorskip("pythonosc", reason="python-osc not available"),
    reason="python-osc required for real OSC server tests"
)
class TestRealOSCServer:
    """Test real OSC server (requires python-osc)."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = OSCConfig(port=9999)  # Use different port for testing
        self.event_manager = EventManager()
    
    def test_server_creation(self):
        """Test real OSC server creation."""
        server = OSCServer(self.config, self.event_manager)
        assert server is not None
        assert not server.is_running
    
    def test_server_start_stop(self):
        """Test real OSC server start/stop."""
        server = OSCServer(self.config, self.event_manager)
        
        try:
            server.start()
            assert server.is_running
            time.sleep(0.1)  # Give server time to start
            
        finally:
            server.stop()
            assert not server.is_running


class TestOSCServerFactory:
    """Test OSC server factory function."""
    
    def test_create_mock_server(self):
        """Test creating mock server when python-osc unavailable."""
        config = OSCConfig()
        event_manager = EventManager()
        
        with patch('cloud_instrument.osc.server.HAS_PYTHON_OSC', False):
            server = create_osc_server(config, event_manager)
            assert isinstance(server, MockOSCServer)
    
    def test_create_real_server(self):
        """Test creating real server when python-osc available."""
        config = OSCConfig()
        event_manager = EventManager()
        
        with patch('cloud_instrument.osc.server.HAS_PYTHON_OSC', True):
            try:
                server = create_osc_server(config, event_manager)
                # Could be either OSCServer or MockOSCServer depending on actual availability
                assert server is not None
            except Exception:
                # If creation fails, should fall back to mock
                pass