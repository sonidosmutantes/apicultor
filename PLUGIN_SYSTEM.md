# Apicultor Plugin System

Apicultor features a modular plugin architecture that allows you to enable or disable components based on your needs. This system provides flexibility, reduces dependencies, and allows for customized deployments.

## Table of Contents

- [Overview](#overview)
- [Available Plugins](#available-plugins)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Plugin Development](#plugin-development)
- [Troubleshooting](#troubleshooting)

## Overview

The plugin system in Apicultor allows you to:

- **Enable/disable modules** through configuration files or environment variables
- **Reduce memory footprint** by loading only needed components
- **Customize deployments** for different environments (development, production, embedded systems)
- **Manage dependencies** automatically between plugins
- **Extend functionality** by developing custom plugins

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Config        │───▶│  Plugin Manager  │───▶│   Enabled       │
│   System        │    │                  │    │   Plugins       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Plugin         │
                    │   Interfaces     │
                    └──────────────────┘
```

## Available Plugins

### Core Plugins

| Plugin | Description | Dependencies | Key Features |
|--------|-------------|--------------|--------------|
| **database** | Sound database access | None | Freesound, RedPanal, local files |
| **constraints** | Mathematical constraints | None | Optimization bounds, tempo sync |
| **analysis** | MIR feature extraction | None | Spectral analysis, descriptors |

### Analysis Plugins

| Plugin | Description | Dependencies | Key Features |
|--------|-------------|--------------|--------------|
| **emotion** | Music emotion analysis | machine_learning | Deep learning emotion detection |
| **machine_learning** | Sound similarity & clustering | None | K-means, SVM, similarity metrics |
| **segmentation** | Audio segmentation | None | Fixed, adaptive, MIR-based |

### Generation Plugins

| Plugin | Description | Dependencies | Key Features |
|--------|-------------|--------------|--------------|
| **state_machine** | Markov chain composition | database | Real-time performance, OSC |
| **sonification** | Data-to-audio conversion | None | Parameter mapping, synthesis |
| **gradients** | Optimization algorithms | None | SGD, attention mechanisms |

## Configuration

### 1. Configuration File

Create an `apicultor_config.json` file:

```json
{
  "plugins": {
    "enabled_modules": [
      "database",
      "constraints", 
      "machine_learning",
      "segmentation"
    ],
    "disabled_modules": [
      "emotion",
      "state_machine"
    ],
    "auto_discover": true,
    "fail_on_plugin_error": false,
    "plugin_configs": {
      "database": {
        "default_provider": "freesound",
        "cache_size": 1000,
        "freesound_api_key": "${APICULTOR_FREESOUND_API_KEY}"
      },
      "machine_learning": {
        "default_algorithm": "kmeans",
        "max_clusters": 10
      },
      "constraints": {
        "tolerance": 1e-8,
        "max_iterations": 1000
      }
    }
  }
}
```

### 2. Environment Variables

```bash
# Enable specific modules
export APICULTOR_ENABLED_MODULES="database,constraints,machine_learning"

# Disable specific modules  
export APICULTOR_DISABLED_MODULES="emotion,state_machine"

# Plugin system settings
export APICULTOR_AUTO_DISCOVER="true"
export APICULTOR_FAIL_ON_PLUGIN_ERROR="false"

# API keys and secrets
export APICULTOR_FREESOUND_API_KEY="your_freesound_api_key"
```

### 3. Programmatic Configuration

```python
from apicultor.config.settings import Settings
from apicultor.core.plugin_manager import PluginConfig

# Create custom plugin configuration
plugin_config = PluginConfig()
plugin_config.enabled_modules = ["database", "constraints"]
plugin_config.disabled_modules = ["emotion"]
plugin_config.plugin_configs = {
    "database": {"default_provider": "local"},
    "constraints": {"tolerance": 1e-6}
}

# Initialize with custom config
from apicultor.core import initialize_plugin_manager
plugin_manager = initialize_plugin_manager(plugin_config)
```

## Usage Examples

### Basic Usage

```python
import apicultor

# Initialize the plugin system
plugin_manager = apicultor.initialize()

# Check what's available and enabled
print("Available plugins:", plugin_manager.list_available_plugins())
print("Enabled plugins:", plugin_manager.list_enabled_plugins())

# Check if specific plugins are enabled
if plugin_manager.is_plugin_enabled('database'):
    print("Database plugin is ready!")
```

### Using Database Plugin

```python
# Get the database plugin
db_plugin = plugin_manager.get_plugin('database')

if db_plugin:
    # Search for sounds
    results = db_plugin.search_sounds("piano", limit=10)
    print(f"Found {len(results)} sounds")
    
    # Get sound by ID
    sound = db_plugin.get_sound_by_id("123456")
    
    # Download sound
    success = db_plugin.download_sound("123456", "/path/to/output.wav")
    
    # List available providers
    providers = db_plugin.list_providers()
    print("Available providers:", providers)
```

### Using Constraints Plugin

```python
import numpy as np

# Get the constraints plugin
constraints_plugin = plugin_manager.get_plugin('constraints')

if constraints_plugin:
    # Apply mathematical constraints
    values = np.array([-1.0, 0.5, 2.0, -0.3, 1.5])
    
    # Remove negative values
    constrained = constraints_plugin.apply_lower_bounds(values)
    print("After lower bounds:", constrained)
    
    # Apply upper bounds
    weights = np.ones_like(values)
    upper_constrained = constraints_plugin.apply_upper_bounds(
        constrained, weights, multiplier=1.0
    )
    
    # Validate constraints
    validation = constraints_plugin.validate_constraints(upper_constrained)
    print("Validation results:", validation)
```

### Dynamic Plugin Management

```python
# Enable a plugin at runtime
success = plugin_manager.enable_plugin('emotion')
if success:
    print("Emotion plugin enabled!")

# Disable a plugin
plugin_manager.disable_plugin('state_machine')

# Reload a plugin (useful for development)
plugin_manager.reload_plugin('machine_learning')

# Get plugins by interface type
from apicultor.core.interfaces import DatabaseInterface
db_plugins = plugin_manager.get_plugins_by_interface(DatabaseInterface)
```

### Using Machine Learning Plugin

```python
# Get machine learning plugin
ml_plugin = plugin_manager.get_plugin('machine_learning')

if ml_plugin:
    # Load audio features (example)
    features = np.random.rand(100, 13)  # 100 samples, 13 features
    
    # Train a clustering model
    model = ml_plugin.train_model(features, algorithm='kmeans', n_clusters=5)
    
    # Make predictions
    predictions = ml_plugin.predict(model, features)
    print("Cluster assignments:", predictions)
```

## Plugin Development

### Creating a Custom Plugin

1. **Implement the Plugin Interface**:

```python
from apicultor.core.interfaces import PluginInterface
from typing import Any, Dict, List

class MyCustomPlugin(PluginInterface):
    def __init__(self):
        self._enabled = False
        self._config = {}
    
    @property
    def name(self) -> str:
        return "my_custom_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "My custom Apicultor plugin"
    
    @property
    def dependencies(self) -> List[str]:
        return ["database"]  # Depends on database plugin
    
    def initialize(self, config: Dict[str, Any]) -> None:
        self._config = config
        # Initialize your plugin here
        self._enabled = True
    
    def shutdown(self) -> None:
        # Clean up resources
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    # Add your custom methods here
    def my_custom_method(self, data):
        """Your custom functionality."""
        return f"Processed: {data}"
```

2. **Register Your Plugin**:

```python
# Register with plugin manager
plugin_manager = apicultor.get_plugin_manager()
plugin_manager._plugin_classes["my_custom_plugin"] = MyCustomPlugin

# Enable your plugin
plugin_manager.enable_plugin("my_custom_plugin")
```

3. **Use Plugin-Specific Interfaces**:

For specialized plugins, implement specific interfaces:

```python
from apicultor.core.interfaces import AudioProcessorInterface
import numpy as np

class MyAudioProcessor(AudioProcessorInterface):
    # Implement all AudioProcessorInterface methods
    def process_audio(self, audio, sample_rate, **kwargs):
        # Your audio processing logic
        return processed_audio
    
    @property
    def supported_formats(self):
        return [".wav", ".mp3", ".ogg"]
    
    @property
    def required_sample_rates(self):
        return []  # Any sample rate
```

### Plugin Configuration Schema

Define configuration schema for your plugins:

```python
# In your plugin's initialize method
def initialize(self, config: Dict[str, Any]) -> None:
    # Validate required configuration
    required_keys = ["api_key", "endpoint"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Set defaults
    self.timeout = config.get("timeout", 30)
    self.max_retries = config.get("max_retries", 3)
    
    # Initialize with config
    self._setup_api_client(config["api_key"], config["endpoint"])
```

## Deployment Scenarios

### Minimal Setup (Analysis Only)

```json
{
  "plugins": {
    "enabled_modules": ["analysis", "constraints"],
    "disabled_modules": ["database", "emotion", "state_machine"]
  }
}
```

### Research Setup (Full ML Pipeline)

```json
{
  "plugins": {
    "enabled_modules": [
      "database", "analysis", "machine_learning", 
      "emotion", "segmentation", "constraints"
    ],
    "disabled_modules": ["state_machine"]
  }
}
```

### Performance Setup (Real-time)

```json
{
  "plugins": {
    "enabled_modules": [
      "database", "state_machine", "sonification"
    ],
    "disabled_modules": ["emotion", "machine_learning"]
  }
}
```

### Embedded/IoT Setup

```json
{
  "plugins": {
    "enabled_modules": ["analysis", "constraints"],
    "disabled_modules": ["database", "emotion", "machine_learning", "state_machine"]
  }
}
```

## Troubleshooting

### Common Issues

1. **Plugin Not Found**:
   ```
   Error: Plugin 'emotion' not found
   ```
   - Check that the plugin name is spelled correctly
   - Verify the plugin is in `enabled_modules` and not in `disabled_modules`
   - Check if auto-discovery is enabled

2. **Dependency Errors**:
   ```
   Error: Dependencies not satisfied for plugin 'emotion'
   ```
   - Enable required dependency plugins first
   - Check the dependency chain in plugin documentation

3. **Configuration Errors**:
   ```
   Error: Missing required config key: 'api_key'
   ```
   - Check plugin-specific configuration requirements
   - Verify environment variables are set correctly

4. **Import Errors**:
   ```
   ImportError: No module named 'some_dependency'
   ```
   - Install required dependencies: `pip install missing-package`
   - Check if the plugin has optional dependencies

### Debug Mode

Enable debug logging to troubleshoot plugin issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
import os
os.environ["APICULTOR_LOG_LEVEL"] = "DEBUG"
```

### Plugin Health Check

```python
# Check plugin status
plugin_manager = apicultor.get_plugin_manager()

for plugin_name in plugin_manager.list_enabled_plugins():
    plugin = plugin_manager.get_plugin(plugin_name)
    print(f"{plugin_name}: {'✓' if plugin.is_enabled else '✗'}")

# Validate configuration
from apicultor.config import get_settings
settings = get_settings()
try:
    settings.validate()
    print("Configuration is valid ✓")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Performance Monitoring

```python
import time

# Monitor plugin initialization time
start_time = time.time()
plugin_manager = apicultor.initialize()
init_time = time.time() - start_time

print(f"Plugin system initialized in {init_time:.2f}s")
print(f"Enabled plugins: {len(plugin_manager.list_enabled_plugins())}")
```

## Best Practices

1. **Configuration Management**:
   - Use environment variables for sensitive data (API keys)
   - Keep configuration files in version control (without secrets)
   - Use different configurations for different environments

2. **Plugin Development**:
   - Follow the interface contracts strictly
   - Handle errors gracefully in plugin methods
   - Implement proper cleanup in shutdown methods
   - Document configuration requirements

3. **Performance**:
   - Only enable plugins you actually use
   - Consider lazy loading for heavy plugins
   - Monitor memory usage with large plugin sets

4. **Security**:
   - Validate all plugin configuration inputs
   - Use secure defaults
   - Regularly update plugin dependencies

5. **Testing**:
   - Test with minimal plugin configurations
   - Verify plugin dependencies work correctly
   - Test error conditions and recovery

## Contributing

To contribute a new plugin to Apicultor:

1. Create your plugin following the interface guidelines
2. Add comprehensive tests
3. Document configuration options
4. Submit a pull request with examples

For questions or support, please open an issue on the Apicultor GitHub repository.
