# coding: utf-8
# python2
# > ipython2

# Granular example (by Pablo Riera: https://github.com/sonidosmutantes/taller-procesamiento-experimentacion/blob/master/freeze/python/Granulator%20and%20PV%20with%20pyo.ipynb)

from pyo import *
s = Server().boot()
#s = Server(audio='jack').boot()
s.start()

audio_file_path = "../samples/984_sample0.wav"

snd = SndTable( audio_file_path )
env = HannTable()
pos = Phasor(freq=snd.getRate()*.25, mul=snd.getSize())
dur = Noise(mul=.001, add=.1)
g = Granulator(snd, env, [1, 1.001], pos, dur, 32, mul=.1).out()

# GUI
# s.gui(locals())

# s.stop()

#MIDI
# names, indexes = pm_get_input_devices()
# print(names,indexes)

# #s = Server(audio='jack').boot()
# s.start()
# s.setMidiInputDevice(3) # Must be called before s.boot()
# s.boot()
# s.start()

# noin = Notein(poly=1,scale=0)

# posx = Port( Midictl(ctlnumber=[78], minscale=0, maxscale=snd.getSize()), 0.02)
# posf = Port( Midictl(ctlnumber=[16], minscale=0, maxscale=snd.getSize()), 0.02)
# porta = Midictl(ctlnumber=[79], minscale=0., maxscale=60.)


posxx = (noin['pitch']-48.)/(96.-48.)*posf+posx
pos = SigTo(posxx)

def set_ramp_time():
    pos.time = porta.get()

tf = TrigFunc(Change(porta), function=set_ramp_time)

pitch = Port(Midictl(ctlnumber=[17], minscale=0.0, maxscale=2.0),0.02)

noisemul = Midictl(ctlnumber=[18], minscale=0.0, maxscale=0.2)

noiseadd = Port(Midictl(ctlnumber=[19], minscale=0.0, maxscale=1.0),0.02)

dur = Noise(mul=noisemul)+noiseadd

g = Granulator(snd, env, pitch*0.1/dur, pos , dur, 16, mul=.3).mix(2).out()