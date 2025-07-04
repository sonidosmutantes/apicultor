# APICultor Cloud Instrument

A modern, real-time sound synthesis system with OSC control and MIR-based sound selection.

## Overview

The Cloud Instrument is a complete refactored and modernized version of the original APICultor cloud instrument. It provides real-time audio synthesis controlled via OSC (Open Sound Control) messages, with intelligent sound selection based on Music Information Retrieval (MIR) descriptors.

### Key Features

- 🎵 **Real-time Audio Synthesis** - SuperCollider, Pyo, or Mock backends
- 🎛️ **OSC Control Interface** - Complete OSC message routing and handling
- 🎹 **MIDI Controller Support** - Yaeltex, Akai Midimix, and generic controllers
- 🔍 **MIR-based Sound Selection** - Search sounds by audio characteristics
- 🔌 **Plugin Architecture** - Modular, extensible component system
- ⚙️ **Modern Configuration** - JSON files with environment variable support
- 🧪 **Comprehensive Testing** - pytest-based test suite with mocks
- 📝 **Full Type Safety** - Complete type annotations for Python 3.8+

## Quick Start

### 1. Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

**Quick install:**
```bash
# Install dependencies
pip install python-osc python-rtmidi librosa ffmpeg-python requests

# Optional: Install freesound-python
pip install "freesound-python @ git+https://github.com/MTG/freesound-python.git"
```

### 2. Configuration

Create a configuration file `cloud_instrument_config.json`:

```json
{
  "osc": {
    "host": "127.0.0.1",
    "port": 9001,
    "enable_logging": true
  },
  "audio": {
    "backend": "supercollider",
    "sample_rate": 44100,
    "buffer_size": 512
  },
  "midi": {
    "enabled": true,
    "virtual_port": true,
    "client_name": "CloudInstrument"
  },
  "database": {
    "default_provider": "freesound",
    "freesound_api_key": "${FREESOUND_API_KEY}"
  },
  "enabled_modules": ["database"]
}
```

For Freesound API access, get your API key from [Freesound APIv2](http://www.freesound.org/apiv2/apply/).

### 3. Run the Application

**Modern version (recommended):**
```bash
python CloudInstrument_modern.py
```

**Legacy version:**
```bash
python CloudInstrument.py
```

**With custom configuration:**
```bash
python CloudInstrument_modern.py -c my_config.json
```

**With command-line overrides:**
```bash
python CloudInstrument_modern.py --osc-port 9002 --audio-backend pyo --no-midi
```

## Architecture

### Modern Architecture (2025 Refactor)

```
cloud_instrument/
├── core/                    # Core application components
│   ├── application.py          # Main app orchestrator
│   ├── config.py              # Configuration management
│   ├── events.py              # Event system
│   └── exceptions.py          # Exception hierarchy
├── osc/                     # OSC system
│   ├── server.py              # OSC server + mock fallback
│   ├── handlers.py            # Message routing & handling
│   └── messages.py            # OSC message types
├── audio/                   # Audio system
│   ├── manager.py             # Audio coordinator
│   ├── backends.py            # Audio server implementations
│   ├── effects.py             # Audio effects
│   └── interfaces.py          # Audio interfaces
├── midi/                    # MIDI system
│   ├── manager.py             # MIDI coordinator
│   ├── controllers.py         # Hardware controller support
│   └── messages.py            # MIDI message handling
└── tests/                   # Modern test suite
    ├── test_config.py
    └── test_osc.py
```

### Key Improvements

| Aspect | Before (Legacy) | After (Modern) |
|--------|----------------|----------------|
| **Python Version** | 2.7 | 3.8+ with type annotations |
| **Architecture** | Monolithic script | Modular, event-driven |
| **Configuration** | Hard-coded | Flexible JSON + environment variables |
| **Error Handling** | Basic | Comprehensive with graceful degradation |
| **Testing** | Minimal | pytest with mocks and fixtures |
| **Documentation** | Limited | Complete with examples |

## Usage

### OSC Messages

The Cloud Instrument responds to various OSC messages:

#### Audio Effects
```
/fx/volume [float]           # Volume control (0.0-1.0)
/fx/pan [float]              # Pan position (-1.0 to 1.0)
/fx/reverb [room_size, damping]  # Reverb parameters
/fx/delay [time, feedback]   # Delay effect
/fx/filter [freq, resonance, type]  # Filter control
```

#### MIR Parameters
```
/mir/tempo [bpm]             # Target tempo
/mir/centroid [hz]           # Spectral centroid target
/mir/duration [seconds]      # Target duration
/mir/hfc [value]             # High frequency content
```

#### Sound Control
```
/sound/search [query]        # Search sound database
/sound/play [id]             # Play sound by ID
/sound/stop                  # Stop playback
```

#### System Control
```
/system/status               # Get system status
/system/shutdown             # Graceful shutdown
```

### MIDI Controllers

#### Supported Controllers

**Yaeltex Controller:**
- Faders 0-7: Channel volumes
- Knob 16: Pan control
- Knobs 17-18: Reverb room/damping
- Knobs 19-20: Delay time/feedback
- Buttons: Play/Stop/Record

**Akai Midimix:**
- Track faders: Volume control
- Upper knobs: Pan, reverb
- Lower knobs: Delay, filter
- Buttons: Mute/Solo/Record per track

**Generic Controllers:**
- Configurable CC and note mappings
- Custom controller support

### Audio Backends

#### SuperCollider
```bash
# Start SuperCollider first
sclang -D apicultor_synth.scd
```

#### Pyo
```bash
# Python-based audio processing
python CloudInstrument_modern.py --audio-backend pyo
```

#### Mock (Testing)
```bash
# No audio hardware required
python CloudInstrument_modern.py --audio-backend mock
```

## User Interfaces

### OpenStageControl UI
Load `ui/apicultor-ui.json` in [OpenStageControl](https://osc.ammd.net/) with OSC receiving port 7000.

![UI ArCiTec](../doc/UI%20ArCiTec.png)

### MIDI Controllers

#### Yaeltex Custom Controller
![Yaeltex Controller](../doc/yaeltex-pre-print-front.png)

Custom [Yaeltex](https://yaeltex.com/en) MIDI controllers provide dedicated hardware control.

![Controller](../doc/controller.jpg)

#### Modes

**MIR Mode:**
![MIR Mode](../doc/modo-mir.png)

**Synth Mode:**
![Synth Mode](../doc/modo-synth.png)

## Audio Processing Chain

```
Input → Effects Chain → Output
        ↓
freeze → vibrato → pan → pitch shift → filters → delay → reverb
```

## Development

### Testing
```bash
# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cloud_instrument --cov-report=html

# Test specific component
pytest tests/test_osc.py -v
```

### Type Checking
```bash
mypy cloud_instrument/
```

### Code Formatting
```bash
black cloud_instrument/
flake8 cloud_instrument/
```

### Command Line Options

```bash
python CloudInstrument_modern.py --help
```

Options include:
- `-c, --config`: Configuration file path
- `--osc-host/--osc-port`: OSC server settings
- `--audio-backend`: Audio backend choice
- `--midi-device`: MIDI device selection
- `--no-midi`: Disable MIDI
- `-v, --verbose`: Debug logging
- `--status`: Show system status
- `--test-osc`: Test OSC functionality

## Environment Variables

Set environment variables for sensitive configuration:

```bash
export FREESOUND_API_KEY="your_api_key_here"
export APICULTOR_LOG_LEVEL="DEBUG"
```

## Troubleshooting

### Common Issues

**OSC not working:**
- Check if python-osc is installed: `pip install python-osc`
- Verify port is not in use: `netstat -an | grep 9001`

**MIDI not working:**
- Install python-rtmidi: `pip install python-rtmidi`
- Check available MIDI devices: `python CloudInstrument_modern.py --status`

**Audio backend issues:**
- SuperCollider: Ensure scsynth/sclang are in PATH
- Pyo: Install pyo: `pip install pyo`
- Use mock backend for testing: `--audio-backend mock`

### Logging

Enable debug logging for troubleshooting:
```bash
python CloudInstrument_modern.py --verbose
```

Or set in configuration:
```json
{
  "logging": {
    "level": "DEBUG",
    "file": "cloud_instrument.log"
  }
}
```

## Integration

### APICultor Plugin System

The Cloud Instrument integrates with the APICultor plugin system:

```python
from cloud_instrument import CloudInstrumentApp

app = CloudInstrumentApp("config.json")
app.run()
```

### External Applications

Send OSC messages from any application:
```python
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 9001)
client.send_message("/fx/volume", 0.8)
```

## Performance

### Raspberry Pi Configuration

For optimal performance on Raspberry Pi:

```bash
# ~/.jackdrc
/usr/bin/jackd -P75 -t2000 -dalsa -dhw:S2 -p4096 -n7 -r44100 -s
```

### Real-time Audio

Configure low-latency audio:
```json
{
  "audio": {
    "sample_rate": 44100,
    "buffer_size": 256,
    "channels": 2
  }
}
```

## License

Free Software shared with GPL v3. See [LICENSE](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

## Support

- Documentation: See [docs](../docs_new/) directory
- Issues: Report on GitHub
- Examples: See [examples](../examples/) directory