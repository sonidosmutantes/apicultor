#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def dyn_constraint_satis(audio, variables, gain):
    assert ((type(variables) is list) and len(variables) == 2)
    #attenuate
    audio[variables[0] < variables[1]] = gain      
    audio[variables[0] > variables[1]] = 1
    return audio

def gate_constraint_satis(audio, variables, gain):
    assert ((type(variables) is list) and len(variables) == 2)
    #close unfeasible
    audio[variables[0] < variables[1]] = 0 
    #keep feasible open
    audio[variables[0] > variables[1]] = 1   
    return audio
