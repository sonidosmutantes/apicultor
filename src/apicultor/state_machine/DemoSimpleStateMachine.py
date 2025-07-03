#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import pykov # Markov chains helpers
import time
import random
import urllib.request, urllib.error, urllib.parse
import OSC
import sys

# RedPanal API
URL_BASE = "http://127.0.0.1:5000" #TODO: get from a config file

# OSC Server
osc_client = OSC.OSCClient()
sc_Port = 57120
sc_IP = '127.0.0.1' #Local SC server
#sc_IP = '10.142.39.109' #Remote server
# Virtual Box: Network device config not in bridge or NAT mode
# Select 'Network host-only adapter' (Name=vboxnet0)
sc_IP = '192.168.56.1' # Remote server is the host of the VM
osc_client.connect( ( sc_IP, sc_Port ) )

# 3 states  (each row must sum 1)
# idle -> no sound
# harmonic -> choose one harmonic sound (or note) from database with a given frec and time?
# inharmonic

T = pykov.Matrix()

T['idle','harmonic'] = .2
T['idle','inharmonic'] = .1
T['idle','idle'] = .7

T['harmonic','idle'] = .2
T['harmonic','inharmonic'] = .1
T['harmonic','harmonic'] = .7

T['inharmonic','idle'] = .9
T['inharmonic','inharmonic'] = .1
#T['inharmonic','inharmonic'] = 0


try:
    T.stochastic() #check
except Exception as e:
    print(e)
    exit(1)


duration = 1 #FIXME: hardcoded
time_bt_states = 1 # (delay within states...)
state = 'idle' #start state

events = 10 # or loop with while(1)
# for i in range(events):
while(1):
      print( state ) # TODO: call the right method for the state here
      if state=='harmonic':
        call = '/list/samples' #gets only wav files because SuperCollider
        response = urllib.request.urlopen(URL_BASE + call).read()
        audioFiles = list()
        for file in response.split('\n'):
            if len(file)>0: #avoid null paths
                audioFiles.append(file)
                # print file
        file_chosen = audioFiles[ random.randint(0,len(audioFiles)-1) ]
        print(("\tPlaying %s"%file_chosen))
        msg = OSC.OSCMessage()
        msg.setAddress("/play")

        #mac os
        msg.append( "/Users/hordia/Documents/apicultor"+file_chosen.split('.')[1]+'.wav' )

        try:
            osc_client.send(msg)
        except Exception as e:
            print(e)
        #TODO: get duration from msg (via API)
        time.sleep(duration)


      state = T.succ(state).choose() #new state
      time_between_notes = random.uniform(0.,2.) #in seconds
      time.sleep(time_between_notes)
      #delay within states
      time.sleep(time_bt_states)
