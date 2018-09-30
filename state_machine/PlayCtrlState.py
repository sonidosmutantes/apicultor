#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pykov # Markov chains helpers
import time
import random
import urllib.request, urllib.error, urllib.parse
import OSC
import sys
import os.path
import json
from pyo import *
import signal
import liblo
from liblo import make_method

from ..mir.db.FreesoundDB import FreesoundDB
from ..mir.db.RedPanalDB import RedPanalDB

import platform
#from __future__ import print_function
from random import random

from math import pow

DATA_PATH = "data"
SAMPLES_PATH = "samples"

state = dict() #global MIR state
state["duration"] = "1." # default value
volume = 1.

# OSC Server
# class OSCServer(liblo.ServerThread):
    # def __init__(self, port):
    #     liblo.ServerThread.__init__(self, port)

# @make_method("/volume", 'f')
def update_volume_callback(path, args):
    global volume
    value = args[0]
    volume = value
    print(("Update volume: %s", value))
    # out = c.mix(2, mul=value).out() #dry output
    s.amp = float(value) # server amplitude

# @make_method("/pitch", 'ff')
def pitch_shift_callback(path, args):
    print(("Received %s %s"%(path,args)))
    global c
    global volume
    try:
        value = args[0]
        state = int(args[1])
        if state==0: return #on release
        #take a reference (now 60 TODO update) and calculate amount of steps shift
        amount = pow(2,(value-60.)/12.)
        # FIXME
        print(("Pitch shift amount %f"%amount))
        c = FreqShift(c, shift=amount, mul=volume)
    except Exception as e:
        print(e)

# @make_method("/retrieve", 'i')
def search_by_mir_state_callback(path, args):
    print(("Received %s %s"%(path,args)))
    global state
    global volume
    if args[0]==1:
        print(("Estado actual %s"%state))
        print(("Estado actual on/off %s"%osc_desc_state))

    try:
        new_state = dict()
        for desc in state:
            if osc_desc_state[desc]==True:
                new_state[ osc_desc_conv[desc]] = state[desc]

        #FIXME: needs two descriptors? duration and other?
        # duration always have a number (minor equal) < value
        new_state["sfx.duration"] = "* TO %s"%new_state["sfx.duration"]

        #TODO: filter state values (search with AND only if they are enabled (on==1))
        print(("Estado MIR freesound a buscar: %s"%new_state))

        file_chosen, author, sound_id  = api.get_one_by_mir(new_state)
        #(needs to wait here?)
        
        print(( os.path.getsize(file_chosen) ))
        if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000: #FIXME: prior remove 'silence' sounds from DB (ETL)
            print(file_chosen)
            log.write(file_chosen+" by "+ author + " - id: "+str(sound_id)+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit. FIXM
            pyo_synth(file_chosen, volume)
            # pyo_synth_noisevc("./wow1.wav", float(args[0])) 
    except Exception as e:
        print(e)
#available descriptors
osc_desc_conv = {
    "content": "content",
    "BPM": "BPM",
    "duration": "sfx.duration",
    "inharmonicity/mean": "sfx.inharmonicity.mean",
    "hfc/mean": "lowlevel.hfc.mean",
    "spectral_centroid/mean": "lowlevel.spectral_centroid.mean",
    "spectral_complexity/mean": "lowlevel.spectral_complexity.mean"
}
osc_descriptors = list()
# for key,value in osc_desc_conv.iteritems():
#     osc_descriptors
osc_descriptors = ["duration", "BPM", "hfc", "spectral_complexity/mean", "spectral_centroid/mean", "inharmonicity" ]
osc_desc_state = dict()
for desc in osc_descriptors:
    osc_desc_state[desc] = False

# @make_method(None, None)
def update_state_fallback(path, args, types, src):
    global state
    global osc_desc_state
    # print("got unknown message '%s' from '%s'" % (path, src.url))
    # for a, t in zip(args, types):
    #     print("argument of type '%s': %s" % (t, a))
    msg = path[1:]
    value = args[0]
    print(("Received %s %s"%(path,args)))

    if msg[-3:]=="/on":
        # print("ON/OFF: %s", value)
        desc = msg[:-3]
        osc_desc_state[desc] = True if value==1 else False
        print(( "%s state %i"%(desc,osc_desc_state[desc]) ))
        return
    if msg in osc_descriptors:
        desc = msg
        state[ desc ] = value
        print(( "MIR state updated! %s: %f"%(desc,value) ))
    print(("Value %s"%value))


# OSC Client (i.e. send OSC to SuperCollider)
# osc_client = OSC.OSCClient()
# sc_Port = 57120
# sc_IP = '127.0.0.1' #Local SC server
#sc_IP = '10.142.39.109' #Remote server
# Virtual Box: Network device config not in bridge or NAT mode
# Select 'Network host-only adapter' (Name=vboxnet0)
# sc_IP = '192.168.56.1' # Remote server is the host of the VM
# osc_client.connect( ( sc_IP, sc_Port ) )


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
    s.setJackAuto(False, False) #some linux bug workaround
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
    print(("\tPlaying %s"%new_file))
    msg = OSC.OSCMessage()
    msg.setAddress("/play")

    #mac os #FIXME
    msg.append( "/Users/hordia/Documents/apicultor"+new_file.split('.')[1]+'.wav' )

    try:
        osc_client.send(msg)
    except Exception as e:
        print(e)
    #TODO: get duration from msg (via API)
    time.sleep(duration)
#external_synth()



def pyo_synth(new_file, dry_value):
    """
        default synth (freeze intent)
    """
    #Phase Vocoder
    sfplay = SfPlayer(new_file, loop=True, mul=dry_value)
    pva = PVAnal(sfplay, size=1024, overlaps=4, wintype=2)
    pvs = PVAddSynth(pva, pitch=1., num=500, first=10, inc=10).mix(2)#.out() 
    # pvs = PVAddSynth(pva, pitch=notes['pitch'], num=500, first=10, inc=10, mul=p).mix(2).out()

    c.setInput(pvs, fadetime=.25)
    # c = c.mix(2).out()
#pyo_synth()

def pyo_synth_noisevc(new_file, dry_value):
    print("noise vocoder synth")
    # First sound - dynamic spectrum.
    spktrm = SfPlayer(new_file, speed=[1,1.001], loop=True, mul=dry_value)

    # Second sound - rich and stable spectrum.
    excite = Noise(0.2)

    # LFOs to modulated every parameters of the Vocoder object.
    lf1 = Sine(freq=0.1, phase=random()).range(60, 100)
    lf2 = Sine(freq=0.11, phase=random()).range(1.05, 1.5)
    lf3 = Sine(freq=0.07, phase=random()).range(1, 20)
    lf4 = Sine(freq=0.06, phase=random()).range(0.01, 0.99)

    voc = Vocoder(spktrm, excite, freq=lf1, spread=lf2, q=lf3, slope=lf4, stages=32)

    c.setInput(voc, fadetime=.25)
    # c = c.mix(2).out()
#pyo_synth_noisevc()

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
        print(("\nBad amount of input arguments\n\t", Usage, "\n"))
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
    except Exception as e:
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
    print(("Using "+api_type+" API"))

    osc_port = config["osc.port"]

    #JSON composition file
    json_data = ""
    try:
        json_comp_file = sys.argv[1] 
        # with open(json_file,'r') as file:
        #     json_data = json.load( file )
        json_data = json.load( open(json_comp_file,'r') )
    except Exception as e:
        print(e)
        print("JSON composition file error.")
        sys.exit(2)

    print("Starting MIR state machine")
    log.write("Starting MIR state machine: "+json_comp_file+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit
    
    # print( json_data['statesArray'][0]['mir'] )


    # # Init state (starts playing!)
    # print("MIR State: "+str(mir_state))
    # mir_state = json_data['statesArray'][0]['mir'][0]
    # file_chosen, autor, sound_id  = api.get_one_by_mir(mir_state)
    # #hardcoded file
    file_chosen, autor, sound_id  = "./Tape Start Electric.wav", "void", "0"

    # print( os.path.getsize(file_chosen) )
    if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000: #FIXME: prior remove 'silence' sounds from DB (ETL)
        print(file_chosen)
        log.write(file_chosen+" by "+ autor + " - id: "+str(sound_id)+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit. FIXME
        pyo_synth(file_chosen, dry_val)
        # pyo_synth_noisevc(file_chosen, dry_val)
    #     #s.gui(locals())

    if 1:
        # create server, listening on port 1234
        try:
            server = liblo.Server(osc_port)
            # server = OSCServer(osc_port)
        except liblo.ServerError as err:
            print(err)
            sys.exit()

        server.add_method("/pitch", 'if', pitch_shift_callback) # register method taking two floats
        server.add_method("/volume", 'f', update_volume_callback) # register method taking a float
        server.add_method("/retrieve", 'i', search_by_mir_state_callback)
        # server.add_method("/on", 'i', on_callback)
        server.add_method(None, None, update_state_fallback) # register a fallback for unhandled messages (any other message)

        # loop and dispatch messages every 100ms
        while True:
            server.recv(100)
            # server.start()
            # input("press enter to quit...\n")


                # Hardcoded sound for each MIR state
                # file_chosen = snd_dict[state]
                # granular_synth(file_chosen)
                # external_synth(file_chosen)


        # time_bt_states = states_dur[ state ]
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

    log.close()
    #end
    global volume
    try:
        value = args[0]
        state = int(args[1])
        if state==0: return #on release
        #take a reference (now 60 TODO update) and calculate amount of steps shift
        amount = pow(2,(value-60.)/12.)
        # FIXME
        print("Pitch shift amount %f"%amount)
        c = FreqShift(c, shift=amount, mul=volume)
    except Exception, e:
        print(e)

# @make_method("/retrieve", 'i')
def search_by_mir_state_callback(path, args):
    print("Received %s %s"%(path,args))
    global state
    global volume
    if args[0]==1:
        print("Estado actual %s"%state)
        print("Estado actual on/off %s"%osc_desc_state)

    try:
        new_state = dict()
        for desc in state:
            if osc_desc_state[desc]==True:
                new_state[ osc_desc_conv[desc]] = state[desc]

        #FIXME: needs two descriptors? duration and other?
        # duration always have a number (minor equal) < value
        new_state["sfx.duration"] = "* TO %s"%new_state["sfx.duration"]

        #TODO: filter state values (search with AND only if they are enabled (on==1))
        print("Estado MIR freesound a buscar: %s"%new_state)

        file_chosen, author, sound_id  = api.get_one_by_mir(new_state)
        #(needs to wait here?)
        
        print( os.path.getsize(file_chosen) )
        if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000: #FIXME: prior remove 'silence' sounds from DB (ETL)
            print(file_chosen)
            log.write(file_chosen+" by "+ author + " - id: "+str(sound_id)+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit. FIXM
            pyo_synth(file_chosen, volume)
            # pyo_synth_noisevc("./wow1.wav", float(args[0])) 
    except Exception, e:
        print(e)
#available descriptors
osc_desc_conv = {
    "content": "content",
    "BPM": "BPM",
    "duration": "sfx.duration",
    "inharmonicity/mean": "sfx.inharmonicity.mean",
    "hfc/mean": "lowlevel.hfc.mean",
    "spectral_centroid/mean": "lowlevel.spectral_centroid.mean",
    "spectral_complexity/mean": "lowlevel.spectral_complexity.mean"
}
osc_descriptors = list()
# for key,value in osc_desc_conv.iteritems():
#     osc_descriptors
osc_descriptors = ["duration", "BPM", "hfc", "spectral_complexity/mean", "spectral_centroid/mean", "inharmonicity" ]
osc_desc_state = dict()
for desc in osc_descriptors:
    osc_desc_state[desc] = False

# @make_method(None, None)
def update_state_fallback(path, args, types, src):
    global state
    global osc_desc_state
    # print("got unknown message '%s' from '%s'" % (path, src.url))
    # for a, t in zip(args, types):
    #     print("argument of type '%s': %s" % (t, a))
    msg = path[1:]
    value = args[0]
    print("Received %s %s"%(path,args))

    if msg[-3:]=="/on":
        # print("ON/OFF: %s", value)
        desc = msg[:-3]
        osc_desc_state[desc] = True if value==1 else False
        print( "%s state %i"%(desc,osc_desc_state[desc]) )
        return
    if msg in osc_descriptors:
        desc = msg
        state[ desc ] = value
        print( "MIR state updated! %s: %f"%(desc,value) )
    print("Value %s"%value)


# OSC Client (i.e. send OSC to SuperCollider)
# osc_client = OSC.OSCClient()
# sc_Port = 57120
# sc_IP = '127.0.0.1' #Local SC server
#sc_IP = '10.142.39.109' #Remote server
# Virtual Box: Network device config not in bridge or NAT mode
# Select 'Network host-only adapter' (Name=vboxnet0)
# sc_IP = '192.168.56.1' # Remote server is the host of the VM
# osc_client.connect( ( sc_IP, sc_Port ) )


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
    s.setJackAuto(False, False) #some linux bug workaround
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
    """
        default synth (freeze intent)
    """
    #Phase Vocoder
    sfplay = SfPlayer(new_file, loop=True, mul=dry_value)
    pva = PVAnal(sfplay, size=1024, overlaps=4, wintype=2)
    pvs = PVAddSynth(pva, pitch=1., num=500, first=10, inc=10).mix(2)#.out() 
    # pvs = PVAddSynth(pva, pitch=notes['pitch'], num=500, first=10, inc=10, mul=p).mix(2).out()

    c.setInput(pvs, fadetime=.25)
    # c = c.mix(2).out()
#pyo_synth()

def pyo_synth_noisevc(new_file, dry_value):
    print("noise vocoder synth")
    # First sound - dynamic spectrum.
    spktrm = SfPlayer(new_file, speed=[1,1.001], loop=True, mul=dry_value)

    # Second sound - rich and stable spectrum.
    excite = Noise(0.2)

    # LFOs to modulated every parameters of the Vocoder object.
    lf1 = Sine(freq=0.1, phase=random()).range(60, 100)
    lf2 = Sine(freq=0.11, phase=random()).range(1.05, 1.5)
    lf3 = Sine(freq=0.07, phase=random()).range(1, 20)
    lf4 = Sine(freq=0.06, phase=random()).range(0.01, 0.99)

    voc = Vocoder(spktrm, excite, freq=lf1, spread=lf2, q=lf3, slope=lf4, stages=32)

    c.setInput(voc, fadetime=.25)
    # c = c.mix(2).out()
#pyo_synth_noisevc()

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

    osc_port = config["osc.port"]

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
    
    # print( json_data['statesArray'][0]['mir'] )


    # # Init state (starts playing!)
    # print("MIR State: "+str(mir_state))
    # mir_state = json_data['statesArray'][0]['mir'][0]
    # file_chosen, autor, sound_id  = api.get_one_by_mir(mir_state)
    # #hardcoded file
    file_chosen, autor, sound_id  = "./Tape Start Electric.wav", "void", "0"

    # print( os.path.getsize(file_chosen) )
    if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000: #FIXME: prior remove 'silence' sounds from DB (ETL)
        print(file_chosen)
        log.write(file_chosen+" by "+ autor + " - id: "+str(sound_id)+"\n") #WARNING: bad realtime practice (writing file) TODO: add to a memory buffer and write before exit. FIXME
        pyo_synth(file_chosen, dry_val)
        # pyo_synth_noisevc(file_chosen, dry_val)
    #     #s.gui(locals())

    if 1:
        # create server, listening on port 1234
        try:
            server = liblo.Server(osc_port)
            # server = OSCServer(osc_port)
        except liblo.ServerError as err:
            print(err)
            sys.exit()

        server.add_method("/pitch", 'if', pitch_shift_callback) # register method taking two floats
        server.add_method("/volume", 'f', update_volume_callback) # register method taking a float
        server.add_method("/retrieve", 'i', search_by_mir_state_callback)
        # server.add_method("/on", 'i', on_callback)
        server.add_method(None, None, update_state_fallback) # register a fallback for unhandled messages (any other message)

        # loop and dispatch messages every 100ms
        while True:
            server.recv(100)
            # server.start()
            # input("press enter to quit...\n")


                # Hardcoded sound for each MIR state
                # file_chosen = snd_dict[state]
                # granular_synth(file_chosen)
                # external_synth(file_chosen)


        # time_bt_states = states_dur[ state ]
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

    log.close()
    #end
