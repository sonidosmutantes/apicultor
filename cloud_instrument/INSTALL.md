# APICultor Cloud Instrument - Installation Guide

Complete installation guide for the modernized Cloud Instrument system.

## Requirements

### System Requirements
- **Python**: 3.8 or higher (Python 2.7 is no longer supported)
- **Operating System**: Linux, macOS, Windows
- **Memory**: 512MB RAM minimum, 1GB recommended
- **Audio Hardware**: Built-in audio or external audio interface

### Hardware Support
- **Raspberry Pi**: 3B+ or 4 recommended
- **MIDI Controllers**: Yaeltex custom controllers, Akai Midimix, generic MIDI devices
- **Audio Interfaces**: JACK-compatible, i2s soundcards (PiHat DAC)

## Quick Installation

### 1. Install Python Dependencies

```bash
# Core dependencies
pip install python-osc python-rtmidi librosa requests

# Audio processing
pip install pyo  # Optional: Python audio engine

# Development tools
pip install pytest mypy black flake8
```

### 2. Install Cloud Instrument

```bash
# Clone repository
git clone https://github.com/sonidosmutantes/apicultor.git
cd apicultor/cloud_instrument

# Install in development mode
pip install -e .
```

### 3. Quick Test

```bash
# Test with mock audio backend
python CloudInstrument_modern.py --audio-backend mock --test-osc
```

## Detailed Installation

### Audio Backends

#### SuperCollider (Recommended)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install supercollider supercollider-server
```

**macOS:**
```bash
brew install supercollider
```

**Raspberry Pi (headless):**
```bash
sudo apt install supercollider-server
# No GUI components needed
```

For custom builds on Raspberry Pi:
```bash
# Follow official guide for RT kernel support
# https://supercollider.github.io/development/building-raspberry
```

#### Pyo Audio Engine

**Ubuntu/Debian:**
```bash
sudo apt install python3-dev libjack-jackd2-dev libportmidi-dev \
    portaudio19-dev liblo-dev libsndfile-dev

pip install pyo
```

**macOS:**
```bash
brew install jack portaudio liblo libsndfile
pip install pyo
```

#### JACK Audio (Optional)

```bash
# Ubuntu/Debian
sudo apt install jackd2 qjackctl

# macOS
brew install jack
```

### MIDI Support

#### Python MIDI Dependencies

```bash
# Core MIDI support
pip install python-rtmidi

# Check installation
python -c "import rtmidi; print('MIDI support OK')"
```

#### Platform-Specific MIDI Setup

**Linux:**
```bash
sudo apt install librtmidi-dev
```

**macOS:**
```bash
brew install rtmidi
```

**Windows:**
```bash
# Use precompiled wheels
pip install python-rtmidi
```

### Sound Database Integration

#### Freesound API

```bash
# Install Freesound client
pip install "freesound-python @ git+https://github.com/MTG/freesound-python.git"

# Get API key from: http://www.freesound.org/apiv2/apply/
export FREESOUND_API_KEY="your_api_key_here"
```

#### Optional: Red Panal API

```bash
# No additional installation required
# Configure endpoint in cloud_instrument_config.json
```

### Audio Processing Tools

#### FFmpeg for Audio Conversion

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
pip install ffmpeg-python
```

**macOS:**
```bash
brew install ffmpeg
pip install ffmpeg-python
```

**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
pip install ffmpeg-python
```

#### Audio Normalization

```bash
pip install ffmpeg-normalize
```

## Hardware-Specific Setup

### Raspberry Pi Configuration

#### Audio Setup (i2s Soundcard)

**Pimoroni pHAT DAC:**
```bash
curl https://get.pimoroni.com/phatdac | bash
```

**Manual i2s Setup:**
```bash
# Add to /boot/config.txt
echo "dtoverlay=hifiberry-dac" | sudo tee -a /boot/config.txt
sudo reboot
```

#### JACK Configuration

```bash
# Create JACK config
cat > ~/.jackdrc << EOF
/usr/bin/jackd -P75 -t2000 -dalsa -dhw:S2 -p4096 -n7 -r44100 -s
EOF

# List audio devices
aplay -l

# Edit config for your specific device
nano ~/.jackdrc
```

#### Performance Optimizations

```bash
# Real-time kernel (optional)
sudo apt install linux-image-rt-amd64

# Audio group permissions
sudo usermod -a -G audio $USER

# System limits
echo "@audio - rtprio 95" | sudo tee -a /etc/security/limits.conf
echo "@audio - memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

### MIDI Controller Setup

#### Yaeltex Controllers

```bash
# Custom SuperCollider extensions (optional)
sudo cp Extensions/* /usr/local/share/SuperCollider/Extensions/
```

#### Generic Controllers

```bash
# List available MIDI devices
python CloudInstrument_modern.py --status
```

## Configuration

### Create Configuration File

```bash
# Create basic config
cat > cloud_instrument_config.json << EOF
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
EOF
```

### Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export FREESOUND_API_KEY="your_api_key_here"
export APICULTOR_LOG_LEVEL="INFO"
export CLOUD_INSTRUMENT_CONFIG="./cloud_instrument_config.json"
```

## Verification

### Test Installation

```bash
# Test core functionality
python CloudInstrument_modern.py --test-osc

# Test with specific backend
python CloudInstrument_modern.py --audio-backend mock --verbose

# Check system status
python CloudInstrument_modern.py --status
```

### Run Test Suite

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cloud_instrument --cov-report=html
```

## Platform-Specific Notes

### Ubuntu/Debian

```bash
# Complete installation script
sudo apt update
sudo apt install python3-pip python3-dev git
sudo apt install supercollider-server jackd2 librtmidi-dev ffmpeg
pip3 install python-osc python-rtmidi librosa pyo pytest
```

### macOS

```bash
# Using Homebrew
brew install python supercollider jack rtmidi ffmpeg
pip3 install python-osc python-rtmidi librosa pyo pytest
```

### Windows

```bash
# Using pip only (audio backends may require additional setup)
pip install python-osc python-rtmidi librosa requests pytest

# For audio backends, see platform documentation:
# SuperCollider: https://supercollider.github.io/
# JACK: https://jackaudio.org/
```

## Troubleshooting

### Common Issues

**Python 2.7 Dependencies:**
The modern version requires Python 3.8+. Legacy Python 2.7 dependencies are no longer supported.

**OSC Not Working:**
```bash
# Check if port is in use
netstat -an | grep 9001

# Test with different port
python CloudInstrument_modern.py --osc-port 9002
```

**MIDI Issues:**
```bash
# List MIDI devices
python -c "import rtmidi; print(rtmidi.MidiIn().get_ports())"

# Use virtual MIDI port
python CloudInstrument_modern.py --midi-device virtual
```

**Audio Backend Issues:**
```bash
# Test with mock backend
python CloudInstrument_modern.py --audio-backend mock

# Check SuperCollider installation
sclang --version

# Check Pyo installation
python -c "import pyo; print('Pyo OK')"
```

### Debug Mode

```bash
# Enable verbose logging
python CloudInstrument_modern.py --verbose

# Use debug configuration
python CloudInstrument_modern.py -c debug_config.json
```

### Performance Issues

**Raspberry Pi:**
```bash
# Increase GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 64

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi-country
```

**Audio Latency:**
```bash
# Reduce JACK buffer size
# Edit ~/.jackdrc: change -p4096 to -p256
```

## Development Setup

### Development Dependencies

```bash
pip install -e ".[dev]"
# Or manually:
pip install pytest mypy black flake8 pytest-cov pytest-mock
```

### IDE Integration

**VS Code:**
```bash
# Install Python extension
# Add to settings.json:
{
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black"
}
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Legacy Version Notes

The old `CloudInstrument.py` script required Python 2.7 and used deprecated dependencies:

- **pyosc** (replaced by python-osc)
- **pyliblo** (replaced by python-osc)
- **python2-specific** packages

These are no longer supported. Use `CloudInstrument_modern.py` for all new installations.

## Migration from Legacy

If upgrading from the old version:

1. **Backup your configuration** and custom SuperCollider code
2. **Install Python 3.8+** and modern dependencies
3. **Update configuration** to JSON format
4. **Test with mock backend** before enabling audio hardware
5. **Update any custom OSC clients** for new message format

## License

Free Software shared with GPL v3. See [LICENSE](LICENSE).

## Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Report on GitHub
- **Examples**: See [examples](../examples/) directory