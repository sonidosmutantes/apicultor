#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pykov # Markov chains helpers
import time
import random
import urllib2
import OSC
import sys
import os.path
import json
from pyo import *
import signal

#TODO: addi/use formal loggin
# import logging
# import logging.handlers

from mir.db.FreesoundDB import FreesoundDB
from mir.db.RedPanalDB import RedPanalDB

import platform
#from __future__ import print_function


DATA_PATH = "data"
SAMPLES_PATH = "samples"

# OSC Server
osc_client = OSC.OSCClient()
sc_Port = 57120
sc_IP = '127.0.0.1' #Local SC server
#sc_IP = '10.142.39.109' #Remote server
# Virtual Box: Network device config not in bridge or NAT mode
# Select 'Network host-only adapter' (Name=vboxnet0)
sc_IP = '192.168.56.1' # Remote server is the host of the VM
osc_client.connect( ( sc_IP, sc_Port ) )


### Signal handler (ctrl+c)
def signal_handler(signal, frame):
    global log
    print('Ctrl+C')
    log.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

### Pyo Sound Server ###

if platform.system() == "Darwin" or platform.system() == "Windows":
    ### Default
    s = Server().boot()
    # s = Server(duplex=0).boot()
    # s = Server(audio='portaudio').boot()
    s = Server().boot()
else: #Linux
    ### JACK ###
    # or export PYO_SERVER_AUDIO=jack (~/.bashrc)
    s = Server(audio='jack')
#    s.setJackAuto(False, False) #some linux bug workaround (not needed with jackd compiled without dbus, when X system is not running)
    s.boot()
    s.setJackAutoConnectOutputPorts(['system:playback_1', 'system:playback_2'])

s.start() #no s.gui(locals())

"""
    ETL of the sound database
    =========================

    Audio normalization performed offline to save realtime resources (raspberry pi implementation)
    see ../helper scripts

    TODO: remove "silence" sounds from db (actually checking the file length)
"""

# sffade = Fader(fadein=0.05, fadeout=1, dur=0, mul=0.5).play()

# Mixer
# 3 outputs mixer, 1 second of amplitude fade time
#mm = Mixer(outs=3, chnls=2, time=1)

dry_val = 1 
wet_val = 0.5 #check which reverb algorithm is using
# dry_val = 0.7
# dry_val = 0.3
a = Sine(freq=10, mul=0.3) #start signal

VOL_ADJUST = 6
c = Clip(a, mul=VOL_ADJUST)
#d = c.mix(2).out() #full dry output
out = c.mix(2).out() #dry output

# Reverb
# b1 = Allpass(out, delay=[.0204,.02011], feedback=0.35) # wet output
# b2 = Allpass(b1, delay=[.06653,.06641], feedback=0.41)
# b3 = Allpass(b2, delay=[.035007,.03504], feedback=0.5)
# b4 = Allpass(b3, delay=[.023021 ,.022987], feedback=0.65)
# c1 = Tone(b1, 5000, mul=0.2).out()
# c2 = Tone(b2, 3000, mul=0.2).out()
# c3 = Tone(b3, 1500, mul=0.2).out()
# c4 = Tone(b4, 500, mul=0.2).out()

#Another reverb
# comb1 = Delay(out, delay=[0.0297,0.0277], feedback=0.65)
# comb2 = Delay(out, delay=[0.0371,0.0393], feedback=0.51)
# comb3 = Delay(out, delay=[0.0411,0.0409], feedback=0.5)
# comb4 = Delay(out, delay=[0.0137,0.0155], feedback=0.73)
# combsum = out + comb1 + comb2 + comb3 + comb4
# all1 = Allpass(combsum, delay=[.005,.00507], feedback=0.75)
# all2 = Allpass(all1, delay=[.0117,.0123], feedback=0.61)
# lowp = Tone(all2, freq=3500, mul=wet_val).out()

#buggy? segmentation fault
"""
    8 delay lines FDN (Feedback Delay Network) reverb, with feedback matrix based upon physical modeling scattering junction of 8 lossless waveguides of equal characteristic impedance.
"""
pan = SPan(out, pan=[.25, .4, .6, .75]).mix(2)
rev = WGVerb(pan, feedback=.65, cutoff=3500, bal=.2)
# rev.out()
gt = Gate(rev, thresh=-24, risetime=0.005, falltime=0.01, lookahead=5, mul=.4)
gt.out() 


# Loads the sound file in RAM. Beginning and ending points
# can be controlled with "start" and "stop" arguments.
# t = SndTable(path)

    # #FIXME: test purposes
    # #hardcoded sound files
    # A_snd = "../samples/1194_sample0.wav"
    # B_snd = "../samples/Solo_guitar_solo_sample1.wav"
    # C_snd = "../samples/Cuesta_caminar_batero_sample3.wav"
    # snd_dict = dict()
    # snd_dict["A"] = A_snd
    # snd_dict["B"] = B_snd
    # snd_dict["C"] = C_snd
    # snd_dict["D"] = C_snd
    # snd_dict["E"] = C_snd
    # snd_dict["F"] = C_snd
    # snd_dict["G"] = C_snd
    # snd_dict["H"] = C_snd

# def freesound_search(api_key="", id=""):
#     call = """curl -H "Authorization: Token %(api_key)s" 'http://www.freesound.org/apiv2/sounds/%(id)s/'"""%locals()
#     response = urllib2.urlopen(call).read()
#     print(response)
# #freesound_search()

def external_synth(new_file):
    """
        Sends OSC
        Sends OSC to external synthesis engine like SuperCollider or pd
    """
    print("\tPlaying %s"%new_file)
    msg = OSC.OSCMessage()
    msg.setAddress("/play")

    #mac os #FIXME
    msg.append( "/Users/hordia/Documents/apicultor"+new_file.split('.')[1]+'.wav' )

    try:
        osc_client.send(msg)
    except Exception,e:
        print(e)
    #TODO: get duration from msg (via API)
    time.sleep(duration)
#external_synth()

def pyo_synth(new_file, dry_value):
    #Phase Vocoder
    sfplay = SfPlayer(new_file, loop=True, mul=dry_value)
    pva = PVAnal(sfplay, size=1024, overlaps=4, wintype=2)
    pvs = PVAddSynth(pva, pitch=1., num=500, first=10, inc=10).mix(2)#.out() 
    # pvs = PVAddSynth(pva, pitch=notes['pitch'], num=500, first=10, inc=10, mul=p).mix(2).out()

    c.setInput(pvs, fadetime=.25)
    # c = c.mix(2).out()
#pyo_synth()

def granular_synth(new_file):
    """
        Granulator sound
    """
    pass
    # snd = SndTable(file_chosen)
    # env = HannTable()
    # # note_in_pitch = 62
    # # posx = Port( Midictl(ctlnumber=[78], minscale=0, maxscale=snd.getSize()), 0.02)
    # # posf = Port( Midictl(ctlnumber=[16], minscale=0, maxscale=snd.getSize()), 0.02)
    # #porta = Midictl(ctlnumber=[79], minscale=0., maxscale=60.)
    # # posxx = (note_in_pitch-48.)/(96.-48.)*posf+posx
    # # pos = SigTo(posxx)
    # # tf = TrigFunc(Change(porta), function=set_ramp_time)
    # # pitch = Port(Midictl(ctlnumber=[17], minscale=0.0, maxscale=2.0),0.02)
    # # noisemul = Midictl(ctlnumber=[18], minscale=0.0, maxscale=0.2)
    # # noiseadd = Port(Midictl(ctlnumber=[19], minscale=0.0, maxscale=1.0),0.02)
    # # dur = Noise(mul=noisemul)+noiseadd
    # pitch = 62
    # dur = 3
    # pos = 1
    # g = Granulator(snd, env, pitch*0.1/dur, pos , dur, 16, mul=.3).mix(2).out()
#granulator_synth()

#TODO: chequear si se usa
def set_ramp_time():
    pos.time = porta.get()
    
Usage = "./StateMachine.py [StateComposition.json]"
if __name__ == '__main__':
      
    if len(sys.argv) < 2:
        print("\nBad amount of input arguments\n\t", Usage, "\n")
        sys.exit(1)
    
    logfile = "apicultor.log"
    try:
        log = open(logfile, "a") #append? or overwrite ('w')
    except:
        print("Log file error")
        sys.exit(2)
    
    # JSON config file
    config = ""
    try:
        config = json.load( open(".apicultor_config.json",'r') )
    except Exception, e:
        print(e)
        print("No json config file or error.")
        sys.exit(3)
    
    api_type = config["api"]
    if api_type=="redpanal":
        db_url = config["RedPanal.org"][0]["url"]
        api = RedPanalDB(db_url)
    elif api_type=="freesound":
        freesound_api_key = config["Freesound.org"][0]["API_KEY"]
        api = FreesoundDB()
        api.set_api_key(freesound_api_key)
    else:
        print("Bad api key config")
        sys.exit(4)
    print("Using "+api_type+" API")

    api.av_conv = config["av_conv"]

    #JSON composition file
    json_data = ""
    try:
        json_comp_file = sys.argv[1] 
        # with open(json_file,'r') as file:
        #     json_data = json.load( file )
        json_data = json.load( open(json_comp_file,'r') )
    except Exception, e:
        print(e)
        print("JSON composition file error.")
        sys.exit(2)

    print("Starting MIR state machine")
    log.write("Starting MIR state machine: "+json_comp_file+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit

    states_dict = dict() # id to name conversion
    states_dur = dict() #states duration
    states_mirdef = dict() #mir state definition
    start_state = json_data['statesArray'][0]['text'] #TODO: add as property (start: True)
    for st in json_data['statesArray']:
        states_dict[ st['id'] ] = st['text'] # 'text' is the name of the state
        try:
            states_mirdef[ st['text'] ] = st['mir'][0]
        except:
            states_mirdef[ st['text'] ] = {"sfx.duration": "* TO 3", "sfx.inharmonicity.mean": "0.1" } #default value
        try:
            states_dur[ st['text'] ] = st['duration'] #default value
        except:
            states_dur[ st['text'] ] = 1. # default duration

    
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
    # state = "A" #start state
    state = start_state
    previous_state = "H"



    #Fixed amount or infinite with while(1 ) ()
    # events = 10 # or loop with while(1)
    # for i in range(events):
    while(1):
        print( "State: %s"%state ) # TODO: call the right method for the state here
        #(optional) change sound in the same state or not (add as json config file)
        if state!=previous_state:
            #retrieve new sound

            # call = '/list/samples' #gets only wav files because SuperCollider
            # response = urllib2.urlopen(URL_BASE + call).read()
            # audioFiles = list()
            # for file in response.split('\n'):
            #     if len(file)>0: #avoid null paths
            #         audioFiles.append(file)
            #         # print file


            mir_state = states_mirdef[ state ]
            print("MIR State: "+str(mir_state))
            file_chosen, autor, sound_id  = api.get_one_by_mir(mir_state)

            print( os.path.getsize(file_chosen) )
            if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000: #FIXME: prior remove 'silence' sounds from DB (ETL)
                print(file_chosen)
                log.write(file_chosen+" by "+ autor + " - id: "+str(sound_id)+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit. FIXME
                pyo_synth(file_chosen, dry_val)

                # Hardcoded sound for each MIR state
                # file_chosen = snd_dict[state]
                # granular_synth(file_chosen)
                # external_synth(file_chosen)


        time_bt_states = states_dur[ state ]
        # time_between_notes = random.uniform(0.,2.) #in seconds
        #time.sleep(time_between_notes)
        #TODO: add random variation time?
        #TODO: transpose all to the same pitch

        # MIDI        
        # notes = Notein(poly=10, scale=1, mul=.5)
        # p = Port(notes['velocity'], .001, .5)

        # # Add inputs to the mixer
        # mm.addInput(voice=new_voice, input=sfplay)
        #mm.addInput(voice=new_voice, input=pvs)

        # Delay within states
        time.sleep(time_bt_states)

        #next state
        previous_state = state
        state = T.succ(state).choose() #new state

        # if state==end_state: break

    log.close()
    #end
