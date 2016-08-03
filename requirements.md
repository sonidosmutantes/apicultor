# Requirements

## Mock RedPanal WebService 

$ sudo pip2 install flask

### Doc

sudo pip2 install flask-autodoc

## MIR

### Essentia (http://essentia.upf.edu/)

$ git clone https://github.com/MTG/essentia.git

You can install those dependencies on a Debian/Ubuntu system from official repositories using the commands provided below:
$ sudo apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev

In order to use python bindings for the library, you might also need to install python-numpy-dev (or python-numpy on Ubuntu) and python-yaml for YAML support in python:
$ sudo apt-get install python-numpy-dev python-numpy python-yaml

./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp --with-gaia

To compile everything you’ve configured:
$ ./waf

To install the C++ library and the python bindings (if configured successfully; you might need to run this command with sudo):
$ sudo ./waf install

## State Machine

$ pip install git+git://github.com/riccardoscalco/Pykov@master #both Python2 and Python3

Note that Pykov depends on numpy and scipy

## OSC

# OSC Service Example
$ sudo pip3 install python-osc

# State Machine Example
$ git clone https://github.com/ptone/pyosc.git
$ cd pyosc && sudo ./setup.py install

## Supercollider

$ sudo apt-get install scide 

## Pre-processing

$ sudo apt-get install ffmpeg

