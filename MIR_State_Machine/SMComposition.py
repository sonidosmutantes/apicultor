#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pykov # Markov chains helpers
import time
import random
import urllib2
import OSC
import sys
import json
from pyo import *

DATA_PATH = "../data"
SAMPLES_PATH = "../samples"

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



Usage = "./StateMachine.py [StateComposition.json]"
if __name__ == '__main__':
      
    if len(sys.argv) < 2:
        print("\nBad amount of input arguments\n\t", Usage, "\n")
        sys.exit(1)

    json_data = ""
    try:
        json_file = sys.argv[1] 
        # with open(json_file,'r') as file:
        #     json_data = json.load( file )
        json_data = json.load( open(json_file,'r') )
    except Exception, e:
        print(e)
        print("JSON file error")
        sys.exit(2)


    states_dict = dict() # id to name conversion
    states_dur = dict() #states duration
    for st in json_data['statesArray']:
        states_dict[ st['id'] ] = st['text']
        try:
            states_dur[ st['text'] ] = st['duration']
        except:
            states_dur[ st['text'] ] = 1. # default duration

    #FIXME: test purposes
    #hardcoded sound files
    A_snd = "../samples/1194_sample0.wav"
    B_snd = "../samples/Solo_guitar_solo_sample1.wav"
    C_snd = "../samples/Cuesta_caminar_batero_sample3.wav"
    snd_dict = dict()
    snd_dict["A"] = A_snd
    snd_dict["B"] = B_snd
    snd_dict["C"] = C_snd
    snd_dict["D"] = C_snd
    snd_dict["E"] = C_snd
    snd_dict["F"] = C_snd
    snd_dict["G"] = C_snd
    snd_dict["H"] = C_snd

    #normalize audio output

    s = Server().boot()
    #s = Server(audio='jack').boot()
    s.start()
    sffade = Fader(fadein=0.05, fadeout=1, dur=0, mul=0.5).play()
    
    sd = states_dict
    T = pykov.Matrix()
    for st in json_data['linkArray']:
        # print( float(st['text']) )
        T[ sd[st['from']], sd[st['to']] ] = float( st['text'] )

    try:
        T.stochastic() #check
    except Exception,e:
        print(e)
        exit(1)

#########################
#FIXME: Time
    # duration = 1 #FIXME: hardcoded (default duration)
    # time_bt_states = 1 # (delay within states...)
#########################
#########################

    # Init conditions
    #state = 'idle' #start state
    state = "A" #start state
    previous_state = "H" #start state
    
    #Fixed amount or infinite with while(1 ) ()
    # events = 10 # or loop with while(1)
    # for i in range(events):
    while(1):
        print( state ) # TODO: call the right method for the state here
        # if state=='harmonic':
        #     call = '/list/samples' #gets only wav files because SuperCollider
        #     response = urllib2.urlopen(URL_BASE + call).read()
        #     audioFiles = list()
        #     for file in response.split('\n'):
        #         if len(file)>0: #avoid null paths
        #             audioFiles.append(file)
        #             # print file
        #     file_chosen = audioFiles[ random.randint(0,len(audioFiles)-1) ]
        #     print("\tPlaying %s"%file_chosen)
        #     msg = OSC.OSCMessage()
        #     msg.setAddress("/play")

        #     #mac os
        #     msg.append( "/Users/hordia/Documents/apicultor"+file_chosen.split('.')[1]+'.wav' )

        #     try:
        #         osc_client.send(msg)
        #     except Exception,e:
        #         print(e)
        #     #TODO: get duration from msg (via API)
        #     time.sleep(duration)

        #(optional) change sound in the same state or not (add as json config file)
        if state!=previous_state:
            #retrieve new sound
            call = '/list/samples' #gets only wav files because SuperCollider
            response = urllib2.urlopen(URL_BASE + call).read()
            audioFiles = list()
            for file in response.split('\n'):
                if len(file)>0: #avoid null paths
                    audioFiles.append(file)
                    # print file
            file_chosen = audioFiles[ random.randint(0,len(audioFiles)-1) ]
            print file_chosen
            file_chosen = "."+file_chosen # path adjustment
        
        time_bt_states = states_dur[ state ]
        # time_between_notes = random.uniform(0.,2.) #in seconds
        #time.sleep(time_between_notes)
        #TODO: add random variation time?

        #TODO: transpose all to the same pitch


        # File to play
        # file_chosen = snd_dict[state]
        # play sound
        # sfplay = SfPlayer(snd_dict[state], speed=1, loop=False, mul=sffade).out()
        sfplay = SfPlayer(file_chosen, loop=True, mul=0.7)
        pva = PVAnal(sfplay, size=1024, overlaps=4, wintype=2)
        pvs = PVAddSynth(pva, pitch=1., num=500, first=10, inc=10).mix(2).out() 
        #delay within states
        time.sleep(time_bt_states)

        #next state
        previous_state = state
        state = T.succ(state).choose() #ne state