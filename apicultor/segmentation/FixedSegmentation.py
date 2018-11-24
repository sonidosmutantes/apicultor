#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import os
import numpy as np
from scipy.io import wavfile

# from smst.utils import audio
from essentia import *
from essentia.standard import *
from scipy.fftpack import *
import time
# from smst.models.sine import scale_frequencies, scale_time, from_audio, to_audio

def fixed_segmentation(filename, segments, options):
    """
        Segmenta con  duración fija (pero posición de inicio random)
        Solo soporta wav files como input
    """
    if not '.wav' in filename:
        raise Exception("random_segmentation only process wav files")

    outputPath = options['outputPath']    
    fixed_dur = options['duration']

    #TODO: check if 'samples' dir exists (if not, create it)
    try:
        sr, wavsignal = wavfile.read(filename)
        pos = int(0)
        durSamples = int(fixed_dur*sr)
        posSamples = int( pos*len(wavsignal) )
        if posSamples+durSamples>len(wavsignal):
            print("El archivo no tiene la duración suficiente")
            return
        for i in range(segments):
            signalOut = wavsignal[pos:pos+durSamples]
            baseName = os.path.splitext(filename)[0].split('/')[-1]
            outputFilename = outputPath+'/'+baseName+'_sample'+str(i)+'.wav'
            wavfile.write(outputFilename,sr,np.array(signalOut, dtype='int16'))
            print("File generated: %s"%outputFilename)
    except Exception as e:
        #TODO: add standard logging output
        print("Error: %s"%e)

#TODO: take files dir as parameter
files_dir = "./wavs/"
segments = 5
ext_filter = ['.wav'] #valid audio files FIXME: convert to wav or support read other formats
sr = 44100
options = dict()
options['outputPath'] = './segmented-samples'
options['duration'] = 5. # 3, 5, 10 seconds, etc
>>>>>>> efc0f016583e7b5e7ff853d54507f3d2ddba6067
for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        if not os.path.splitext(f)[1] in ext_filter:
                    continue
        audio_input = subdir+'/'+f
        print( "Processing %s"%audio_input )
        fixed_segmentation( audio_input, segments, options)
