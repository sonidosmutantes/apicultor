#! /usr/bin/env python3
# -*- coding: utf-8 -*-
  
import random
import os

from ..Sonification import write_file
from soundfile import read
import time

def experimental_random_segmentation(audio_input, segments, options, sr):
    """
		(branch mir-dev en Sonidos Mutantes)
        Segmenta con valores aleatorios según opciones
    """

    filename = audio_input
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

    sinusoidal_model_anal = SineModelAnal()
    sinusoidal_model_synth = SineModelSynth()
    outputPath = options['outputPath']    
    min_dur,max_dur = options['duration']

    try:
        x = read(audio_input)[0]
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
                print("Creating samples directory")
                time.sleep(4) 
            outputFilename = outputPath+'/'+baseName+'_sample'+str(i)+'.wav'
            write_file(outputFilename,signalOut,sr)
            print(("File generated: %s"%outputFilename))
            time.sleep(1)
    except Exception as e:
        print(("Error: %s"%e))

#TODO: take files dir as parameter
files_dir = "data"
segments = 5
ext_filter = ['.wav', '.mp3', '.ogg', '.undefined'] #valid audio files 
sr = 44100
options = dict()
options['outputPath'] = 'samples'
options['duration'] = (0.1, 5) # min, max (de 100ms a 5 seg)
for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        if not os.path.splitext(f)[1] in ext_filter:
                    continue
        audio_input = subdir+'/'+f
        print(( "Processing %s"%audio_input ))
        experimental_random_segmentation( audio_input, segments, options, sr)
