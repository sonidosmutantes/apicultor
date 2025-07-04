"""
APICultor Cloud Instrument.

A modern, real-time sound synthesis system with OSC control and MIR-based sound selection.
"""

__version__ = "2.0.0"
__author__ = "APICultor Team"

from .core.application import CloudInstrumentApp
from .core.config import CloudInstrumentConfig
from .osc.server import OSCServer

__all__ = [
    "CloudInstrumentApp",
    "CloudInstrumentConfig", 
    "OSCServer"
]