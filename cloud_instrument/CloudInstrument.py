#! /usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This program receives osc messages to define a sound state based in MIR descriptors like HFC, BPM, duration, spectral centroid and others.

* Realtime synthesis using Pyo engine or via Supercollider
* OSC messages to set MIR state are received in port 9001 (by default) or can be set in the config file
* API Setup: Freesound, custom, redpanal
* Configuration file: .config.json 

Example of config file:
{ 
  "api": "freesound",
  "Freesound.org": [
      { "API_KEY": ""
      }
   ]
}
"""

from __future__ import print_function

import sys
import os.path
import json
import signal
import logging

from time import sleep


sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '../'))

# import urllib2
from enum import Enum
from OSCServer import *
#from synth.PyoAudioServer import PyoAudioServer # disabled by now
from synth.SuperColliderServer import SupercolliderServer
from mir.db.FreesoundDB import FreesoundDB
from mir.db.RedPanalDB import RedPanalDB
from mir.MIRState import MIRState
# from control.MIDI import MIDI # TODO: make it optional (less dependencies)


class ErrorCode(Enum):
    OK = 0
    NO_CONFIG = 1
    BAD_ARGUMENTS = 3
    BAD_CONFIG = 4
    SERVER_ERROR = 5
    NOT_IMPLEMENTED_ERROR = 6

def signal_handler(signal, frame):
    """ Signal handler (Ctrl+c) """
    global logging
    logging.debug('Ctrl+C')
    sys.exit(ErrorCode.OK.value)
signal.signal(signal.SIGINT, signal_handler)

Usage = "./CloudInstument.py"
if __name__ == '__main__':
    logging.basicConfig(filename='instrumento_nube.log',level=logging.DEBUG)
    #logging.basicConfig(filename='instrumento_nube.log',level=logging.ERROR)
    # TODO: add timestamp
    logging.getLogger().addHandler(logging.StreamHandler()) #console std output
    logging.info('Inicio')

    # if len(sys.argv) < 2: 
    #     logging.error("\nBad amount of input arguments\n\t", Usage, "\n")
    #     sys.exit(ErrorCode.BAD_ARGUMENTS.value)
    
    # JSON config file
    config = ""
    try:
        config = json.load( open(".config.json",'r') )
    except Exception, e:
        logging.error(e)
        logging.error("No json config file or error. Write one called .config.json")
        sys.exit(ErrorCode.NO_CONFIG.value)
    
    # API set up
    api = None
    try:
        api_type = config["api"]
        if api_type=="redpanal":
            db_url = config["RedPanal.org"][0]["url"]
            api = RedPanalDB(db_url)
        elif api_type=="freesound":
            freesound_api_key = config["Freesound.org"][0]["API_KEY"]
            api = FreesoundDB()
            api.set_api_key(freesound_api_key)
        else:
            logging.error("Bad api key config")
            sys.exit(ErrorCode.BAD_CONFIG.value)
    except:
        api_type = "redpanal" #default API (local sounds)
    logging.info("Using "+api_type+" API")

    sound_synth = "pyo" #default synth engine
    try:
        sound_synth = config["sound.synth"]
    except:
        pass
    print("Sound synth: %s"%sound_synth)

    osc_port = 9001 #default osc port
    try:
        osc_port = config["osc.port"]
    except:
        pass

    """
    Load an automatic JSON composition file (default)
        json_data = ""
        try:
            logging.debug("Composition json file load and init")
            json_comp_file = sys.argv[1] 
            json_data = json.load( open(json_comp_file,'r') )
            logging.debug( json_data )
        except Exception, e:
            logging.error(e)
            logging.error("JSON composition file error.")
            #WARNING: bad realtime practice (check if it is writing file. Instead, add to a memory buffer and write before exit
            # logging.debug("Starting MIR state machine")
            try:
                logging.debug("Starting MIR state machine: "+json_comp_file+"\n")
            except:
                pass
            # sys.exit(ErrorCode.BAD_ARGUMENTS.value) #not required
    """

    # OSC server, listening on configured port (9001 by default)
    # receiving configuration for MIR State (based in descriptors and expressed as a json file)
    try:
        logging.debug("OSC server init")
        osc_server = OSCServer(osc_port)
        osc_server.start()
    except liblo.ServerError as err:
        logging.error(err)
        sys.exit(ErrorCode.SERVER_ERROR.value)

    # Sound synthesis
    try:
        if sound_synth=="pyo":
            audio_server = PyoAudioServer()
            pyo_server = audio_server.start()
        elif sound_synth=="supercollider":
            audio_server = SupercolliderServer()
            sc_IP = '127.0.0.1' #Local SC server
            # sc_IP = '10.142.39.109' #Remote server
            sc_Port = 57120
            sc_server = audio_server.start(sc_IP, sc_Port)
        else:
            logging.error("Not yet implemented")
            sys.exit(ErrorCode.NOT_IMPLEMENTED_ERROR.value)
    except Exception, e:
        logging.error(e)
        logging.error("Pyo audio synthesis server error")
        sys.exit(ErrorCode.SERVER_ERROR.value)

    
    # osc class callback set up
    osc_server.state = MIRState()
    osc_server.audio_server = audio_server
    audio_server.api = api
    audio_server.logging = logging
    

    # Server loop
    if sound_synth=="pyo":
        pyo_server.gui(locals()) # gui instance (only for pyo)
    else:
        while True:
            sleep(5) # sin esto en raspberry pi el consumo del cpu se va al 100%
            #sleep(1) # sin esto en raspberry pi el consumo del cpu se va al 100%
            pass
