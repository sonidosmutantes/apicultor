#! /usr/bin/env python
# -*- coding: utf-8 -*-
  
import random
import os
import numpy as np
from smst.utils import audio
from essentia import *
from essentia.standard import *
from scipy.fftpack import *
import time
from smst.models.sine import scale_frequencies, scale_time, from_audio, to_audio

#TODO: freeze effect, time stretching
def random_segmentation(filename, segments, options, sr):
    """
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
        print essentia.log.info("Error: %s"%e)
#random_segmentation()

files_dir = "data"
segments = 5
sr = 44100
options = dict()
options['outputPath'] = 'samples'
options['duration'] = (0.1, 5) # min, max (de 100ms a 5 seg)
for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        audio_input = subdir+'/'+f
        print( "Processing %s"%audio_input )
        random_segmentation( audio_input, segments, options, sr)
#
