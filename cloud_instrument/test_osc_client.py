#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OSC Client Test for Cloud Instrument.

Test script to send OSC messages to the Cloud Instrument server.
"""

import time
import sys
from pythonosc.udp_client import SimpleUDPClient

def test_osc_communication():
    """Test OSC communication with the Cloud Instrument."""
    
    # Create OSC client
    client = SimpleUDPClient("127.0.0.1", 9001)
    
    print("Testing OSC communication with Cloud Instrument...")
    print("Make sure CloudInstrument.py is running with OSC enabled!")
    print("-" * 50)
    
    # Test volume control
    print("Sending volume control message...")
    client.send_message("/fx/volume", 0.75)
    time.sleep(0.5)
    
    # Test pan control
    print("Sending pan control message...")
    client.send_message("/fx/pan", -0.3)
    time.sleep(0.5)
    
    # Test reverb control
    print("Sending reverb control message...")
    client.send_message("/fx/reverb", 0.4, 0.6)
    time.sleep(0.5)
    
    # Test MIR parameters
    print("Sending MIR tempo message...")
    client.send_message("/mir/tempo", 120.0)
    time.sleep(0.5)
    
    print("Sending MIR spectral centroid message...")
    client.send_message("/mir/centroid", 1500.0)
    time.sleep(0.5)
    
    print("Sending MIR duration message...")
    client.send_message("/mir/duration", 3.5)
    time.sleep(0.5)
    
    print("Sending MIR HFC message...")
    client.send_message("/mir/hfc", 0.8)
    time.sleep(0.5)
    
    # Test sound search
    print("Sending sound search message...")
    client.send_message("/sound/search", "piano")
    time.sleep(0.5)
    
    # Test system status
    print("Requesting system status...")
    client.send_message("/system/status")
    time.sleep(0.5)
    
    # Test unknown message
    print("Sending unknown message...")
    client.send_message("/test/unknown", "hello", 42)
    time.sleep(0.5)
    
    print("-" * 50)
    print("OSC test messages sent!")
    print("Check the Cloud Instrument console for received messages.")

if __name__ == "__main__":
    try:
        test_osc_communication()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure python-osc is installed: pip install python-osc")
        print("Make sure Cloud Instrument is running on port 9001")
        sys.exit(1)