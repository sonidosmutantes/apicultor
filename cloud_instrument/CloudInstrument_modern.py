#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Cloud Instrument for APICultor.

A real-time sound synthesis system with OSC control and MIR-based sound selection.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add current directory to Python path to enable local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the modern application
from core.application import CloudInstrumentApp
from core.config import CloudInstrumentConfig
from core.exceptions import CloudInstrumentError

logger = logging.getLogger(__name__)


def setup_basic_logging() -> None:
    """Setup basic logging before config is loaded."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_default_config(config_path: Path) -> None:
    """Create a default configuration file.
    
    Args:
        config_path: Path to create config file at
    """
    config = CloudInstrumentConfig()
    config.save_to_file(config_path)
    print(f"Created default configuration at: {config_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Modern Cloud Instrument for APICultor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run with default configuration
  %(prog)s -c config.json            # Run with specific config file
  %(prog)s --create-config           # Create default config file
  %(prog)s --osc-port 9002           # Override OSC port
  %(prog)s --audio-backend pyo       # Use Pyo audio backend
  %(prog)s --no-midi                 # Disable MIDI
  %(prog)s --verbose                 # Enable debug logging
        """
    )
    
    # Configuration options
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    # OSC options
    parser.add_argument(
        '--osc-host',
        default='127.0.0.1',
        help='OSC server host (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--osc-port',
        type=int,
        default=9001,
        help='OSC server port (default: 9001)'
    )
    
    # Audio options
    parser.add_argument(
        '--audio-backend',
        choices=['supercollider', 'pyo', 'mock'],
        help='Audio backend to use'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        help='Audio sample rate'
    )
    
    # MIDI options
    parser.add_argument(
        '--midi-device',
        help='MIDI input device name'
    )
    
    parser.add_argument(
        '--no-midi',
        action='store_true',
        help='Disable MIDI input'
    )
    
    # Logging options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (debug) logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path'
    )
    
    # Plugin options
    parser.add_argument(
        '--plugins',
        nargs='*',
        help='Plugins to enable (overrides config)'
    )
    
    # Status and testing
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--test-osc',
        action='store_true',
        help='Test OSC server and exit'
    )
    
    return parser.parse_args()


def override_config_from_args(config: CloudInstrumentConfig, args: argparse.Namespace) -> None:
    """Override configuration with command line arguments.
    
    Args:
        config: Configuration to modify
        args: Parsed arguments
    """
    # OSC overrides
    if args.osc_host:
        config.osc.host = args.osc_host
    if args.osc_port:
        config.osc.port = args.osc_port
    
    # Audio overrides
    if args.audio_backend:
        from core.config import AudioBackend
        config.audio.backend = AudioBackend(args.audio_backend)
    if args.sample_rate:
        config.audio.sample_rate = args.sample_rate
    
    # MIDI overrides
    if args.no_midi:
        config.midi.enabled = False
    if args.midi_device:
        config.midi.input_device = args.midi_device
    
    # Logging overrides
    if args.verbose:
        config.logging.level = "DEBUG"
    if args.log_file:
        config.logging.file = str(args.log_file)
    
    # Plugin overrides
    if args.plugins is not None:
        config.enabled_modules = args.plugins


def test_osc_functionality(config: CloudInstrumentConfig) -> bool:
    """Test OSC functionality.
    
    Args:
        config: Configuration to use for testing
        
    Returns:
        True if OSC test passed
    """
    try:
        print(f"Testing OSC server on {config.osc.host}:{config.osc.port}...")
        
        from core.events import EventManager
        from osc.server import create_osc_server
        
        event_manager = EventManager()
        osc_server = create_osc_server(config.osc, event_manager)
        
        # Start server
        osc_server.start()
        
        # Test sending a message
        import time
        time.sleep(0.5)  # Give server time to start
        
        try:
            osc_server.send_message(
                config.osc.host, 
                config.osc.port, 
                "/system/status"
            )
            print("✓ OSC message sent successfully")
        except Exception as e:
            print(f"✗ Failed to send OSC message: {e}")
            return False
        
        # Stop server
        osc_server.stop()
        
        print("✓ OSC test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ OSC test failed: {e}")
        return False


def show_system_status(config: CloudInstrumentConfig) -> None:
    """Show system status.
    
    Args:
        config: Configuration to use
    """
    print("=== Cloud Instrument System Status ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Check dependencies
    print("Dependencies:")
    
    # Check python-osc
    try:
        import pythonosc
        try:
            version = pythonosc.__version__
        except AttributeError:
            version = "available"
        print(f"✓ python-osc: {version}")
    except ImportError:
        print("✗ python-osc: not available")
    
    # Check rtmidi
    try:
        import rtmidi
        print(f"✓ python-rtmidi: available")
    except ImportError:
        print("✗ python-rtmidi: not available")
    
    # Check pyo
    try:
        import pyo
        print(f"✓ pyo: available")
    except ImportError:
        print("✗ pyo: not available")
    
    # Check plugin system
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from apicultor.core.plugin_manager import PluginManager
        print("✓ APICultor plugin system: available")
    except ImportError:
        print("✗ APICultor plugin system: not available")
    
    print()
    print("Configuration:")
    print(f"OSC: {config.osc.host}:{config.osc.port}")
    print(f"Audio backend: {config.audio.backend.value}")
    print(f"MIDI enabled: {config.midi.enabled}")
    print(f"Enabled modules: {config.enabled_modules}")


def main() -> int:
    """Main entry point."""
    try:
        # Setup basic logging
        setup_basic_logging()
        
        # Parse arguments
        args = parse_arguments()
        
        # Handle create-config option
        if args.create_config:
            config_path = args.config or Path("cloud_instrument_config.json")
            create_default_config(config_path)
            return 0
        
        # Load configuration
        try:
            app = CloudInstrumentApp(args.config)
            config = app.config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return 1
        
        # Override config with command line args
        override_config_from_args(config, args)
        
        # Reconfigure logging with final settings
        config.setup_logging()
        
        # Handle status option
        if args.status:
            show_system_status(config)
            return 0
        
        # Handle OSC test option
        if args.test_osc:
            success = test_osc_functionality(config)
            return 0 if success else 1
        
        # Run the application
        logger.info("Starting Cloud Instrument...")
        app.run()
        
        logger.info("Cloud Instrument stopped")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except CloudInstrumentError as e:
        logger.error(f"Cloud Instrument error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())