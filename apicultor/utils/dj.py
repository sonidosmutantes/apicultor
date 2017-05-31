#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from ..machine_learning.cache import memoize
from ..segmentation.DoSegmentation import do_segmentation
import numpy as np
from random import choice, randint
from ..utils.algorithms import *
from librosa.effects import time_stretch
import sklearn
import librosa
import scipy

seconds_to_indices = lambda times: np.int32(times * 44100)  

@memoize
def NMF(stft, spectrum, n_sources):
    """
    Sound source separation using NMF
    :param stft: the short-time Fourier Transform of the signal
    :param n_sources: the number of sources                                                                         
    :returns:                                                                                                         
      - Ys: sources
    """
    print("Computing approximations")
    transformer = sklearn.decomposition.NMF(n_components=n_sources)                                  
    H = transformer.fit_transform(spectrum.T).T
    H = transformer.transform(spectrum.T).T
    W = transformer.components_.T
    idx = np.argsort(W, axis = -1)
    W = np.sort(W, axis = -1)
    H = H[idx]
    print("Reconstructing signals")
    Ys = list(scipy.outer(W[:,i], H[i])*np.exp(1j*np.angle(stft)) for i in range(n_sources))
    print("Saving sound arrays")
    ys = [scipy.fftpack.ifft(Y, 1024) for Y in Ys][0].size
    del Ys
    return ys

def overlapped_intervals(interv):   
    """
    Create sections of intervals
    :param interv: list of indices                                                                     
    :returns:                                                                                                         
      - steps: steps from an index to the other
    """                       
    steps = []                                 
    for i in range(1, len(interv)):      
        steps.append([interv[i-1], interv[i]])
    return np.array(steps)

def get_coordinate(index, cls, decisions):  
    return decisions[int(index), cls]

def distance_velocity(audio, coordinate):
    return np.min((np.max([coordinate/len(audio), 0.36]), 10))

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
def crossfade(audio1, audio2, slices):
    """ 
    Apply crossfading to 2 audio tracks. The fade function is randomly applied
    :param audio1: your first signal 
    :param audio2: your second signal
    :param slices: slices of intervals                                                                            
    :returns:                                                                                                         
      - crossfaded audio
    """
    def fade_out(audio):  
        dbs = 20 * np.log10(abs(audio))
        thres = max(dbs)             
        db_steps = np.arange(abs(thres), 120)
        start = 0
        try:
            sections = int(len(dbs)/len(db_steps))
        except Exception as e:
            return audio
        i = 0                            
        while (start + len(db_steps)) < len(dbs):
            dbs[start:sections + start] -= db_steps[i]
            start += sections
            i += 1 
        if dbs.argmin() == 0:
            dbs = dbs[::-1]
        faded = 10 ** (dbs * 0.05)
        faded[audio < 0] *= -1  
        return faded
    def fade_in(audio):  
        dbs = 20 * np.log10(abs(audio))
        try:
            thres = max(dbs)
        except Exception as e:
            return audio
        dbs = dbs[::-1]            
        db_steps = np.arange(abs(thres), 120)
        start = 0
        try:
            sections = int(len(dbs)/len(db_steps))
        except Exception as e:
            return audio
        i = 0                            
        while (start + len(db_steps)) < len(dbs):
            dbs[start:sections + start] -= db_steps[i]
            start += sections
            i += 1  
        if dbs.argmin() != 0:
            dbs = dbs[::-1]
        faded = 10 ** (dbs * 0.05) 
        faded[audio < 0] *= -1  
        return  faded 
    amp1 = np.nonzero(librosa.zero_crossings(audio1))[-1]
    amp2 = np.nonzero(librosa.zero_crossings(audio2))[-1] 
    amp1 = amp1[librosa.util.match_events(slices[0], amp1)]
    amp2 = amp2[librosa.util.match_events(slices[1], amp2)]
    a = []
    for i in range(len(amp1)):
        a.append(list(audio1[slice(amp1[i][0], amp1[i][1])]))
    a_rev = []
    for i in range(len(amp2)):
        a_rev.append(list(audio2[slice(amp2[i][0], amp2[i][1])]))
    if choice([0,1]) == 0:
        amp1=  fade_out(np.concatenate(a))
        amp2=  fade_in(np.concatenate(a_rev))
    else:
        amp2 = fade_in(np.concatenate(a_rev))
        amp1 = fade_out(np.concatenate(a))
    size = min([len(amp1), len(amp2)])
    result = amp1[:size] + amp2[:size] 
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
    proportion = list(range(len(audio)))[0:len(audio):int(len(audio)/16)]
    song = sonify(audio, 44100)
    prop = overlapped_intervals(proportion)
    audio = song.NoiseAdder(audio) #create noisy part for vinyl in turntable
    audio = song.IIR(audio, 22050, 'lowpass')
    audio = song.IIR(audio, 30, 'lowpass')
    def hand_move(audio, rev_audio, prop, slices): #simulation of hand motion in a turntable
        if (coordinate == False) or (coordinate == None):               
            forwards = np.concatenate([speedx(audio[i[0]:i[1]], randint(2,3)) for i in prop])          
            backwards = np.concatenate([speedx(rev_audio[i[0]:i[1]], randint(2,3)) for i in prop])    
        else:
            factor = distance_velocity(audio, coordinate)
            forwards = np.concatenate([speedx(audio[i[0]:i[1]], factor) for i in prop]) 
            backwards = np.concatenate([speedx(audio[i[0]:i[1]], factor) for i in prop])                        
        cf = crossfade(forwards, backwards, slices)
        return cf
    rev_audio = audio[::-1]
    hand_a = lambda: hand_move(audio, rev_audio, prop, slices = [prop, prop])
    hand_b = lambda: hand_move(audio, rev_audio, prop, slices = [prop[:12], prop[:12]])
    hand_c = lambda: hand_move(audio, rev_audio, prop, slices = [prop[1:], prop])
    hand_d = lambda: hand_move(audio, rev_audio, prop, slices = [prop[1:12], prop[:12]])
    hand_e = lambda: hand_move(audio, rev_audio, prop, slices = [prop, prop[:2]])
    hand_f = lambda: hand_move(audio, rev_audio, prop, slices = [prop[:12], prop[2:12]])
    hand_g = lambda: hand_move(audio, rev_audio, prop, slices = [prop[:2], prop[2:]])
    hand_h = lambda: hand_move(audio, rev_audio, prop, slices = [prop[3:12], prop[3:12]])
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
    dur = int(len(audio)/44100)
    if not dur >= 8:
        try:                                                      
            iterations = choice(list(range(int(dur / 2)))) + 1
        except IndexError as e: #audio is too short to be scratched
            pass
        finally:
            return audio
    else:
        iterations = choice(list(range(int(dur / 8)))) + 1
    for i in range(iterations):
        sound = do_segmentation(audio_input = np.float32(audio), audio_input_from_filename = False, audio_input_from_array = True, sec_len = choice(list(range(2,6))), save_file = False) 
        scratches = scratch(sound, coordinate)
        samples_len = len(scratches)
        position = choice(list(range(dur))) * 44100
        audio = np.concatenate([audio[:position - samples_len], scratches, audio[position + samples_len:]])
    return audio
