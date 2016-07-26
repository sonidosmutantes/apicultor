#! /usr/bin/env python
# -*- coding: utf-8 -*-
  
import random
import os
import numpy as np
from scipy.io import wavfile

def random_segmentation(filename, segments, options):
    """
        Segmenta con valores aleatorios según opciones
        Solo soporta wav files como input
    """
    if not '.wav' in filename:
        raise Exception("random_segmentation only process wav files")

    outputPath = options['outputPath']    
    min_dur,max_dur = options['duration']

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
#random_segmentation()

files_dir = "data"
segments = 5
options = dict()
options['outputPath'] = 'samples'
options['duration'] = (0.1, 5) # min, max (de 100ms a 5 seg)
ext_filter = ['.wav'] #valid audio files FIXME: convert to wav or support read other formats
for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        if os.path.splitext(f)[1] in ext_filter:
            audio_input = subdir+'/'+f
            print( "Processing %s"%audio_input )
            random_segmentation( audio_input, segments, options)
#
