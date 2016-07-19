#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pykov # Markov chains helpers
import time
import random

# 3 states  (each row must sum 1)
# idle -> no sound
# harmonic -> choose one harmonic sound (or note) from database with a given frec and time?
# inharmonic

T = pykov.Matrix()

T['idle','harmonic'] = .4
T['idle','inharmonic'] = .1
T['idle','idle'] = .5

T['harmonic','idle'] = .2
T['harmonic','inharmonic'] = .1
T['harmonic','harmonic'] = .7

T['inharmonic','idle'] = .9
T['inharmonic','inharmonic'] = .1
#T['inharmonic','inharmonic'] = 0


try:
    T.stochastic() #check
except Exception,e:
    print(e)
    exit(1)


events = 10 # or loop with while(1)
state = 'idle' #start state
for i in range(events):
      print( state ) # TODO: call the right method for the state here
      state = T.succ(state).choose() #new state
      time_between_notes = random.uniform(0.,2.) #in seconds
      time.sleep(time_between_notes)

