#! /usr/bin/env python
# -*- coding: utf-8 -*-
  
import random
import os
import numpy as np
from scipy.io import wavfile

from smst.utils import audio
from essentia import *
from essentia.standard import *
from scipy.fftpack import *
import time
from smst.models.sine import scale_frequencies, scale_time, from_audio, to_audio

#TODO: make MONO samples? (freeze effect)
def random_segmentation(filename, segments, options):
    """
        Segmenta con valores aleatorios según opciones
        Solo soporta wav files como input
    """
    if not '.wav' in filename:
        raise Exception("random_segmentation only process wav files")

    outputPath = options['outputPath']    
    min_dur,max_dur = options['duration']

    #TODO: check if 'samples' dir exists (if not, create it)
    try:
        sr, wavsignal = wavfile.read(filename)
        for i in range(segments):
            while(1):
                pos = random.uniform(0.,1.) #posición en el archivo normalizada    
                dur = random.uniform(min_dur,max_dur) 
                durSamples = dur*sr
                posSamples = int( pos*len(wavsignal) )
                if posSamples+durSamples<len(wavsignal):
                    break

            signalOut = wavsignal[pos:pos+durSamples]
            baseName = os.path.splitext(filename)[0].split('/')[-1]
            outputFilename = outputPath+'/'+baseName+'_sample'+str(i)+'.wav'
            wavfile.write(outputFilename,sr,np.array(signalOut, dtype='int16'))
            print("File generated: %s"%outputFilename)
    except Exception, e:
        #TODO: add standard logging output
        print("Error: %s"%e)

#Del branch mir-dev
#TODO: freeze effect, time stretching
def experimental_random_segmentation(filename, segments, options, sr):
    """
		(del branch mir-dev)
        Segmenta con valores aleatorios según opciones
    """
    sinusoidal_model_anal = SineModelAnal()
    sinusoidal_model_synth = SineModelSynth()
    outputPath = options['outputPath']    
    min_dur,max_dur = options['duration']

    try:
        x = MonoLoader(filename = audio_input)()
        for i in range(segments):
            while(1):
                pos = random.uniform(0.,2.) #posición en el archivo normalizada    
                dur = random.uniform(min_dur,max_dur) 
                durSamples = dur*sr
                posSamples = int( pos*len(x) )
                if posSamples+durSamples<len(x):
                    break

            signalOut = x[pos:pos+durSamples]
            baseName = os.path.splitext(filename)[0].split('/')[-1]
            if not os.path.exists(outputPath):                         
                os.makedirs(outputPath)                                
                print "Creating samples directory"
                time.sleep(4) 
            outputFilename = outputPath+'/'+baseName+'_sample'+str(i)+'.wav'
            audio.write_wav(signalOut,sr,outputFilename)
            print("File generated: %s"%outputFilename)
            time.sleep(1)
    except Exception, e:
        print("Error: %s"%e)
#experimental_random_segmentation()

#TODO: take files dir as parameter
files_dir = "data"
segments = 5
ext_filter = ['.wav'] #valid audio files FIXME: convert to wav or support read other formats
sr = 44100
options = dict()
options['outputPath'] = 'samples'
options['duration'] = (0.1, 5) # min, max (de 100ms a 5 seg)
for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        if not os.path.splitext(f)[1] in ext_filter:
                    continue
        audio_input = subdir+'/'+f
        print( "Processing %s"%audio_input )
        random_segmentation( audio_input, segments, options)
        #random_segmentation( audio_input, segments, options, sr)
