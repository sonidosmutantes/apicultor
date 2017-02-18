#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from cache import memoize
from DoSegmentation import *
import numpy as np
from random import choice, randint
from essentia.standard import LowPass, NoiseAdder, Windowing, FFT
from librosa.effects import time_stretch
import librosa
import scipy
from smst.utils.math import to_db_magnitudes, from_db_magnitudes

@memoize
def NMF(stft, n_sources):
    """
    Sound source separation using NMF
    :param stft: the short-time Fourier Transform of the signal
    :param n_sources: the number of sources
    :param n: the length of the signal                                                                          
    :returns:                                                                                                         
      - Ys: sources
    """
    print "Computing approximations"
    W, H = librosa.decompose.decompose(np.abs(stft), n_components=n_sources, sort=True)
    print "Reconstructing signals"
    Ys = list(scipy.outer(W[:,i], H[i])*np.exp(1j*np.angle(stft)) for i in xrange(n_sources))
    print "Saving sound arrays"
    ys = [librosa.istft(Y) for Y in Ys]
    del Ys
    return ys

def overlapped_intervals(interv):                          
    steps = []                                 
    for i in xrange(1, len(interv)):      
        steps.append([interv[i-1], interv[i]])
    return np.array(steps)

def get_coordinate(index, cls, decisions):  
    return decisions[index, cls]

def distance_velocity(audio, coordinate):
    return np.min((np.max([coordinate/len(audio), 0.36]), 10))

@memoize
def speedx(sound_array, factor):
    """
    Multiplies the sound's speed by a factor
    :param sound_array: your input sound 
    :param factor: your speed up factor                                                                              
    :returns:                                                                                                         
      - faster sound
    """
    indices = np.round( np.arange(0, len(sound_array), factor, dtype = np.float) )
    indices = indices[indices < len(sound_array)]
    return sound_array[np.int32(indices)]

#apply crossfading into scratching method
@memoize
def crossfade(audio1, audio2):
    """ 
    Apply crossfading to 2 audio tracks. The fade function is randomly applied
    :param audio1: your first signal 
    :param audio2: your second signal                                                                            
    :returns:                                                                                                         
      - crossfaded audio
    """
    def fade_out(audio):  
        dbs = to_db_magnitudes(audio)
        thres = dbs.max()                
        db_steps = np.arange(abs(dbs.max()), 130)
        start = 0
        try:
            sections = len(dbs)/len(db_steps)
        except Exception, e:
            return audio
        i = 0                            
        while (start + len(db_steps)) < len(dbs):
            dbs[start:sections + start] -= db_steps[i]
            start += sections
            i += 1         
        return audio * from_db_magnitudes(dbs)
    def fade_in(audio):  
        dbs = to_db_magnitudes(audio)[::-1]
        thres = dbs.max()                
        db_steps = np.arange(abs(dbs.max()), 120)
        start = 0
        try:
            sections = len(dbs)/len(db_steps)
        except Exception, e:
            return audio
        i = 0                            
        while (start + len(db_steps)) < len(dbs):
            dbs[start:sections + start] -= db_steps[i]
            start += sections
            i += 1 
        dbs = dbs[::-1]      
        return audio * from_db_magnitudes(dbs)           
    if choice([0,1]) == 0:
        amp1=  fade_out(audio1)
        amp2=  fade_in(audio2)      
    else:
        amp1 = fade_in(audio2) 
        amp2 = fade_out(audio1)       
    result = amp1 + amp2
    return 0.5 * result / result.max() 

#scratch your records
def scratch(audio, coordinate = False):
    """ 
    This function performs scratching effects to sound arrays 
    :param audio: the signal you want to scratch
    :param coordinate: to set this to True, a decision variable must be given (a distance) to use a velocity given a learned distance
    :returns:                                                                                                         
      - scratched signal
    """
    proportion = len(audio)/16
    if 0 > proportion:
        return audio
    audio = LowPass(cutoffFrequency = 30)(audio)
    audio = NoiseAdder(level = -70)(audio) #create noisy part for vinyl in turntable 
    def hand_move(audio, rev_audio): #simulation of hand motion in a turntable
        if (coordinate == False) or (coordinate == None):               
            factors = [randint(2,3), randint(2,3)]           
            forwards = speedx(audio, factors[0])             
            backwards = speedx(np.array(rev_audio), factors[1])  
        else:
            factor = distance_velocity(audio, coordinate)
            forwards = speedx(audio, factor)
            backwards = speedx(rev_audio, factor)
        if len(forwards) > len(backwards):
            cf = crossfade(forwards[len(forwards) - len(forwards[:len(backwards)]):], backwards)
            return np.concatenate([forwards[:len(forwards) - len(forwards[:len(backwards)])], cf])
        else:
            cf = crossfade(forwards, backwards[len(backwards) - len(backwards[:len(forwards)]):])
            return np.concatenate([forwards[:len(forwards) - len(forwards[:len(backwards)])], cf])
    rev_audio = audio[::-1]
    hand_a = lambda: hand_move(audio, rev_audio)
    hand_b = lambda: hand_move(audio[:proportion * 12], rev_audio[:proportion*12])
    hand_c = lambda: hand_move(audio[proportion:], rev_audio)
    hand_d = lambda: hand_move(audio[proportion:proportion*12], rev_audio[:proportion*12])
    hand_e = lambda: hand_move(audio, rev_audio[:proportion*2])
    hand_f = lambda: hand_move(audio[:proportion*12], rev_audio[proportion*2:proportion*12])
    hand_g = lambda: hand_move(audio[:proportion*2], rev_audio[proportion*2:])
    hand_h = lambda: hand_move(audio[proportion*3:proportion*12], rev_audio[proportion*3:proportion*12])
    movs = [hand_a, hand_b, hand_c, hand_d, hand_e, hand_f, hand_g, hand_h]
    return np.concatenate([choice(movs)()])

def scratch_music(audio, coordinate = False):
    """ 
    Enclosing function that performs DJ scratching and crossfading to a signal. First of all an arbitrary ammount of times to scratch the audio is chosen to segment the sound n times and also to arbitrarily decide the scratching technique 
    :param audio: the signal you want to scratch 
    :param coordinate: to set this to True, a decision variable must be given (a distance) to use a velocity given a learned distance    
    :returns:                                                                                                         
      - scratched recording
    """
    dur = int(Duration()(audio))
    if not dur >= 8:
        try:                                                      
            iterations = choice(range(int(Duration()(audio) / 2))) + 1
        except IndexError: #audio is too short to be scratched
            return audio
    else:
        iterations = choice(range(int(Duration()(audio) / 8))) + 1
    for i in xrange(iterations):
        sound = do_segmentation(audio_input = np.float32(audio), audio_input_from_filename = False, audio_input_from_array = True, sec_len = choice(range(2,3)), save_file = False) 
        scratches = scratch(sound, coordinate)
        samples_len = len(scratches)
        position = choice(range(int(Duration()(np.float32(audio))))) * 44100
        audio = np.concatenate([audio[:position - samples_len], scratches, audio[position + samples_len:]])
    return audio
