#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def dyn_constraint_satis(audio, variables, gain):
    assert ((type(variables) is list) and len(variables) == 2)
    audio[variables[0] < variables[1]] = gain  #noise gating: anything below a threshold is silenced     
    audio[variables[0] > variables[1]] = 1
    return audio
