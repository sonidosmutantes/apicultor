#! /usr/bin/env python
# -*- coding: utf-8 -*-

from smst.utils import audio
import numpy as np
import essentia
from essentia.standard import *
import os


def do_segmentation(audio_input):

    x = MonoLoader(filename = audio_input)()

    frames_duration = Duration()

    frame_size = 4096

    hop_size = 1024
 
    segments = [frames_duration(frame) for frame in FrameGenerator(x, frameSize=frame_size, hopSize=hop_size)]

    for segment in segments:                                           
        sample = segment*44100 
        output = x[:sample*60] #extend duration of segment                                            
        baseName = os.path.splitext(audio_input)[0].split('/')[-1]     
        outputFilename = 'samples'+'/'+baseName+'_sample'+'.wav'                                                              
        audio.write_wav(output,44100,outputFilename)
        print("File generated: %s"%outputFilename)
	break  

files_dir = "data/bajo/"
ext_filter = ['.wav'] 
for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        if os.path.splitext(f)[1] in ext_filter:
            audio_input = subdir+'/'+f
            do_segmentation( audio_input)
