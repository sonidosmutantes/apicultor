# Installing OSC Dependencies for Cloud Instrument

## The pyliblo Installation Issue

The `pyliblo` package requires system-level dependencies and Cython compilation, which can be complex to install. Here are several solutions:

## Solution 1: Install System Dependencies First (Recommended)

### macOS (using Homebrew)
```bash
# Install liblo system library
brew install liblo

# Install Cython
pip install Cython

# Now install pyliblo
pip install pyliblo
```

### Ubuntu/Debian
```bash
# Install liblo development libraries
sudo apt-get update
sudo apt-get install liblo-dev

# Install Cython
pip install Cython

# Now install pyliblo
pip install pyliblo
```

### Alternative: Use conda
```bash
# Conda has pre-compiled binaries
conda install -c conda-forge pyliblo
```

## Solution 2: Use python-osc (Alternative OSC Library)

`python-osc` is a pure Python OSC implementation that's easier to install:

```bash
pip install python-osc
```

## Solution 3: Mock OSC for Development

For development and testing without OSC hardware, we can create a mock OSC implementation.

## Solution 4: Use the Cloud Instrument Without OSC

The modernized Cloud Instrument already works without OSC dependencies, as demonstrated by our tests.

## Quick Setup Commands

### Option A: Try system installation (macOS)
```bash
brew install liblo
pip install Cython pyliblo
```

### Option B: Use alternative OSC library
```bash
pip install python-osc
```

### Option C: Skip OSC for now
The Cloud Instrument works without OSC - just run:
```bash
python3 CloudInstrument.py
```

## Updating Cloud Instrument for python-osc

If you choose `python-osc`, I can update the Cloud Instrument to use it instead of `pyliblo`.