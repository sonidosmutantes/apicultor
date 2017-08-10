#! /usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Recibe mensajes por OSC para definir un estado sonoro en base a descriptores MIR
Sintetiza sonido utilizando Pyo
"""

from __future__ import print_function

import sys
import os.path
import json
import signal
import logging

# import urllib2
# from random import random
from enum import Enum
# from math import pow

from OSCServer import *
from synth.PyoAudioServer import PyoAudioServer
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

Usage = "./.py [state.json]"
if __name__ == '__main__':
	logging.basicConfig(filename='instrumento_nube.log',level=logging.DEBUG)
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
		logging.error("No json config file or error.")
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
		osc_port = config["sound.synth"]
	except:
		#Opción: enviar osc a SuperCollider
		pass

	osc_port = 9001 #default osc port
	try:
		osc_port = config["osc.port"]
	except:
		pass

	# # JSON composition file (default)
	# json_data = ""
	# try:
	# 	logging.debug("Composition json file load and init")
	# 	json_comp_file = sys.argv[1] 
	# 	json_data = json.load( open(json_comp_file,'r') )
	# 	logging.debug( json_data )
	# except Exception, e:
	# 	logging.error(e)
	# 	logging.error("JSON composition file error.")
	# 	#WARNING: bad realtime practice (check if it is writing file. Instead, add to a memory buffer and write before exit
	# 	# logging.debug("Starting MIR state machine")
	# 	try:
	# 		logging.debug("Starting MIR state machine: "+json_comp_file+"\n")
	# 	except:
	# 		pass
	# 	# sys.exit(ErrorCode.BAD_ARGUMENTS.value) #not required


	# OSC server, listening on configured port (9001 by default)
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
			pass