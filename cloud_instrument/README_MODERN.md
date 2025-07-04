# Cloud Instrument - Modern Version

## Overview

The modernized Cloud Instrument is a complete rewrite of the original CloudInstrument.py using the current Apicultor architecture. It provides real-time sound synthesis controlled via OSC messages, with MIR descriptors driving sound selection and processing.

## Key Features

- **Modern Plugin Architecture**: Uses Apicultor's plugin system for modular functionality
- **Type Safety**: Full type annotations using Python 3.8+ features
- **Improved Configuration**: JSON-based configuration with environment variable support
- **Better Error Handling**: Comprehensive error handling and logging
- **Graceful Shutdown**: Proper resource cleanup on exit
- **SuperCollider Integration**: Updated SuperCollider server communication

## Architecture Changes

### Original vs Modern

| Aspect | Original | Modern |
|--------|----------|---------|
| Python Version | Python 2.7 | Python 3.8+ |
| Architecture | Monolithic | Plugin-based |
| Type Safety | None | Full type annotations |
| Configuration | Basic JSON | Environment variables + JSON |
| Error Handling | Basic | Comprehensive with proper logging |
| Database API | Direct imports | Plugin system |
| Shutdown | Abrupt | Graceful cleanup |

## Configuration

### Modern Configuration File (.config.json)

```json
{
  "api": "freesound",
  "osc": {
    "port": 9001
  },
  "sound": {
    "synth": "supercollider"
  },
  "supercollider": {
    "ip": "127.0.0.1",
    "port": 57120
  },
  "plugins": {
    "enabled_modules": ["database", "analysis", "sonification", "constraints"],
    "disabled_modules": ["segmentation", "state_machine"],
    "plugin_configs": {
      "database": {
        "default_provider": "freesound",
        "freesound_api_key": "${APICULTOR_FREESOUND_API_KEY}",
        "cache_ttl": 3600
      },
      "analysis": {
        "sample_rate": 44100,
        "frame_size": 2048,
        "hop_size": 512
      }
    }
  },
  "logging": {
    "level": "INFO",
    "file": "cloud_instrument.log"
  }
}
```

### Environment Variables

Set your API keys as environment variables:

```bash
export APICULTOR_FREESOUND_API_KEY="your_freesound_api_key"
export APICULTOR_REDPANAL_API_KEY="your_redpanal_api_key"
```

## Usage

### Starting the Cloud Instrument

```bash
# Using default configuration
python3 CloudInstrument.py

# Using custom configuration file
python3 CloudInstrument.py my_config.json
```

### Plugin System Integration

The modern version integrates with Apicultor's plugin system:

```python
# The instrument automatically initializes configured plugins
instrument = CloudInstrument()
instrument.start()

# Plugins are accessible through the plugin manager
database_plugin = instrument.plugin_manager.get_plugin("database")
analysis_plugin = instrument.plugin_manager.get_plugin("analysis")
```

### OSC Interface

Send OSC messages to control the instrument:

```python
import liblo

# Connect to the instrument
target = liblo.Address(9001)

# Send MIR state parameters
liblo.send(target, "/mir/tempo", 120.0)
liblo.send(target, "/mir/spectral_centroid", 1500.0)
liblo.send(target, "/mir/duration", 3.5)
```

## Dependencies

### Required

- Python 3.8+
- Apicultor (current version with plugin system)
- SuperCollider (for audio synthesis)

### Optional

- liblo (for OSC communication)
- pyliblo (Python OSC bindings)
- Pyo (alternative synthesis engine - not yet implemented in modern version)

### Installation

```bash
# Install Apicultor
cd /path/to/apicultor
pip install -e .

# Install OSC dependencies
pip install pyliblo

# For Ubuntu/Debian
sudo apt-get install liblo-dev
```

## Migration from Original

### API Changes

| Original | Modern | Notes |
|----------|--------|-------|
| `from mir.db.FreesoundDB import FreesoundDB` | Plugin system | Use database plugin |
| `from mir.MIRState import MIRState` | Plugin system | Use analysis plugin |
| Direct API calls | Plugin interface | All APIs through plugins |
| Global state | Class-based | Better encapsulation |

### Configuration Migration

Old format:
```json
{
  "api": "freesound",
  "Freesound.org": [{"API_KEY": "key"}]
}
```

New format:
```json
{
  "api": "freesound",
  "plugins": {
    "enabled_modules": ["database"],
    "plugin_configs": {
      "database": {
        "freesound_api_key": "${APICULTOR_FREESOUND_API_KEY}"
      }
    }
  }
}
```

## Error Codes

The modern version uses structured error codes:

- `0` - OK
- `1` - No configuration file
- `3` - Bad arguments
- `4` - Bad configuration
- `5` - Server error
- `6` - Not implemented
- `7` - Plugin error

## Logging

Enhanced logging with multiple outputs:

- Console output for immediate feedback
- File logging for persistent records
- Structured log messages with timestamps
- Configurable log levels

## Performance Improvements

- **Memory Management**: Better resource cleanup
- **CPU Usage**: Optimized main loop for Raspberry Pi
- **Network**: Improved error handling for network operations
- **Startup Time**: Faster initialization with lazy loading

## Future Enhancements

- **Async Support**: Add async/await for better concurrency
- **Pyo Integration**: Restore Pyo synthesis engine support
- **MIDI Support**: Enhance MIDI controller integration
- **Web Interface**: Add web-based control interface
- **Docker Support**: Containerized deployment option

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Apicultor is properly installed
2. **OSC Not Working**: Check liblo installation
3. **SuperCollider Connection**: Verify SC server is running
4. **Plugin Errors**: Check plugin configuration in config file

### Debug Mode

Enable debug logging:

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

## Examples

See the `examples/` directory for:

- Basic configuration files
- OSC control scripts
- SuperCollider patches
- MIDI controller mappings

## Contributing

When contributing to the Cloud Instrument:

1. Follow the modern Apicultor architecture
2. Use type annotations
3. Add comprehensive error handling
4. Update tests and documentation
5. Maintain backward compatibility where possible