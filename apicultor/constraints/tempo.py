#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import librosa

def same_time(x, y):                                                   
    x_tempo = librosa.beat.beat_track(x)[0]                           
    y_tempo = librosa.beat.beat_track(y)[0]                           
    if x_tempo < y_tempo:                                             
        y = librosa.effects.time_stretch(y, x_tempo/y_tempo)           
        if y.size < x.size:                                            
            y = np.resize(y, x.size)                                   
        else:                                                         
            x = np.resize(x, y.size)                                  
    else:                                                             
        x = librosa.effects.time_stretch(x, y_tempo/x_tempo)          
        if x.size < y.size:                                            
            x = np.resize(x, y.size)                                   
        else:                                                          
            y = np.resize(y, x.size)                                   
    return x, y 
