# coding: utf-8
# python2
# > ipython2

# Granular example

from pyo import *
s = Server().boot()
#s = Server(audio='jack').boot()
s.start()

audio_file_path = "984_sample0.wav"

snd = SndTable( audio_file_path )

env = HannTable()
pos = Phasor(freq=snd.getRate()*.25, mul=snd.getSize())
dur = Noise(mul=.001, add=.1)
g = Granulator(snd, env, [1, 1.001], pos, dur, 32, mul=.1).out()

# GUI
# s.gui(locals())

# s.stop()
