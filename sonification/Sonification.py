#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from ..utils.algorithms import *
import os
import sys
import json
import numpy as np
from scipy.io.wavfile import write
import logging
from soundfile import read
from subprocess import call

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  

def write_file(filename, fs, data):           
    write(filename+'.wav', fs, data) 
    call(['ffmpeg', '-i', filename+'.wav', filename+'.ogg', '-y'])
    call(['rm', '-f', filename+'.wav'])

def hfc_onsets(audio):
    """
    Find onsets in music based on High Frequency Content
    :param audio: the input signal                                                                         
    :returns:                                                                                                         
      - hfcs = onsets locations in seconds
    """
    song = sonify(audio, 44100)
    hfcs = []                          
    for frame in song.FrameGenerator():
        song.window()    
        song.Spectrum()    
        hfcs.append(song.hfc())   
    hfcs /= max(hfcs) 
    song.hfcs = hfcs
    fir = firwin(11, 1.0 / 8, window = "hamming")
    song.filtered = np.convolve(hfcs, fir, mode="same")
    song.climb_hills() 
    return np.array([i for i, x in enumerate(song.Filtered) if x > 0]) * song.N

def normalize(signal):
    maximum_normalizing = np.max(np.abs(signal))/-1 
    normalized = np.true_divide(signal,maximum_normalizing)
    return normalized

def mir_sonification(input_filename, inputSoundFile, tag_dir, data):
    input_signal, sampleRate = read(inputSoundFile)

    input_signal = mono_stereo(input_signal)

    retriever = sonify(input_signal, sampleRate)
    retriever.sonify_music()

    descriptors_dir = (tag_dir+'/'+'descriptores') #descriptors directory of tag

    #sound recording with tempo marker

    tempo_dir = (tag_dir+'/'+'tempo')

    if not os.path.exists(tempo_dir):                         
           os.makedirs(tempo_dir)                                
           print("Creando directorio para marcado de tempo")                                            
                                              
    print ("Saving File with tempo beeps") 
    write_file(tempo_dir + '/' + os.path.splitext(input_filename)[0] + 'tempo', retriever.fs, retriever.tempo_onsets + input_signal)

    #spectral centroid of sound recording

    centroid_dir = (tag_dir+'/'+'centroid')

    if not os.path.exists(centroid_dir):                         
           os.makedirs(centroid_dir)                                
           print("Creando directorio para paso de banda de centroide")

    cutoffHz = data['lowlevel.spectral_centroid.mean']
    print ("Filtering Signal")
    signal_centroid = retriever.IIR(input_signal, [float(cutoffHz)-300, float(cutoffHz)], 'bandpass')
    write_file(centroid_dir + '/' + os.path.splitext(input_filename)[0] + 'centroid', sampleRate, normalize(signal_centroid))

    #mfccs of sound recording

    mfcc_dir = (tag_dir+'/'+'mfcc')

    if not os.path.exists(mfcc_dir):                         
           os.makedirs(mfcc_dir)                                
           print("Creando directorio para paso de banda de mfcc")

    print ("Generating Signal according to Mel bands mean")
    write_file(mfcc_dir + '/' + os.path.splitext(input_filename)[0] + 'mfcc', sampleRate, retriever.mfcc_outputs)

    #inharmonicity of sound recording

    inharmonicity_dir = (tag_dir+'/'+'inharmonicity')

    if not os.path.exists(inharmonicity_dir):                         
           os.makedirs(inharmonicity_dir)                                
           print("Creando directorio para filtrado por inarmonia")

    signal_inharmonicity = retriever.AllPass(retriever.f0, retriever.f0 / (1 + float(data['sfx.inharmonicity.mean'])))
    print ("Filtering Signal according to Inharmonicity mean")
    write_file(inharmonicity_dir + '/' + os.path.splitext(input_filename)[0] + 'inharmonicity', 44100, signal_inharmonicity)

    #dissonance of sound recording

    dissonance_dir = (tag_dir+'/'+'dissonance')

    if not os.path.exists(dissonance_dir):                         
           os.makedirs(dissonance_dir)                                
           print("Creando directorio para escucha de disonancia")

    dissonant_f = retriever.f0 + 2.27*(pow(retriever.f0, 0.4777))/(1 + float(data['lowlevel.dissonance.mean']))
    print ("Filtering Signal according to Dissonance mean")
    signal_dissonance = retriever.IIR(input_signal, [retriever.f0, dissonant_f], 'bandpass')
    write_file(dissonance_dir + '/' + os.path.splitext(input_filename)[0] + 'dissonance', sampleRate, normalize(signal_dissonance))

    #loudness sound recording

    loudness_dir = (tag_dir+'/'+'loudness')

    if not os.path.exists(loudness_dir):                         
           os.makedirs(loudness_dir)                                
           print("Creando directorio para escucha de loudness")                                          

    loudness = float(data['loudness.level.mean'])
    loud_sound = loudness * input_signal
    print ("Saving Loud Sound")
    write_file(loudness_dir + '/' + os.path.splitext(input_filename)[0] + 'loudness', sampleRate, normalize(loud_sound)) 

    #sound recording based on valleys

    valleys_dir = (tag_dir+'/'+'valleys')

    if not os.path.exists(valleys_dir):                         
           os.makedirs(valleys_dir)                                
           print("Creando directorio para escucha de contraste espectral basado en valle espectral")                                           
 
    print ("Saving recording Contrast")
    write_file(valleys_dir + '/' + os.path.splitext(input_filename)[0] + 'contrast', sampleRate, retriever.contrast_outputs) 

    #sound recording with hfc marker

    hfc_dir = (tag_dir+'/'+'hfc')

    if not os.path.exists(hfc_dir):                         
           os.makedirs(hfc_dir)                                
           print("Creando directorio para marcado de contenido de frecuencia alta") 

    output = retriever.hfc_locs
    print ("Saving File with hfc marks") 
    write_file(hfc_dir + '/' + os.path.splitext(input_filename)[0] + 'hfc', sampleRate, output + input_signal) 

    #sound recording with attack marker

    attack_dir = (tag_dir+'/'+'attack')

    if not os.path.exists(attack_dir):                         
           os.makedirs(attack_dir)                                
           print("Creando directorio para marcado de ataque")                                              

    output = retriever.attacks
    print ("Saving File with attack marker") 
    write_file(attack_dir + '/' + os.path.splitext(input_filename)[0] + 'attack', sampleRate, output + input_signal)     

Usage = "./Sonification.py [FILES_DIR]"

def main():
    if len(sys.argv) < 2:
        print("\nNeed tag dir\n", Usage, "\n")
        sys.exit(1)

    try:
        files_dir = sys.argv[1]
        descriptors_dir = files_dir+'/descriptores/' 

        if not os.path.exists(files_dir):                             
            raise IOError("Must download sounds")    

        for subdir, dirs, files in os.walk(files_dir):    
            for f in files:    
                tag_dir = subdir    
                input_filename = f    
                audio_input = subdir+'/'+f    
                print( audio_input )    
                with open(descriptors_dir+f.split('.')[0]+'.json') as mir_data:                             
                    data = json.load(mir_data)                        
                mir_sonification(input_filename, audio_input, tag_dir, data)                     
    except Exception as e:
        logger.exception(e)
        sys.exit(1)

if __name__ == '__main__': 
    main()



