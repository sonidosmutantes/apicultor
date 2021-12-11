#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from ..machine_learning.cache import memoize
import numpy as np
from soundfile import read
from ..sonification.Sonification import write_file
from random import choice
from ..utils.algorithms import *
import os

def do_segmentation(audio_input, audio_input_from_filename = True, audio_input_from_array = False, sec_len = 6, save_file = True):

    lenght = int(sec_len) * 10

    if audio_input_from_filename == True:                                           
        #x = read(audio_input)[0]
        x = MonoLoader(filename = audio_input)()
        #print(audio_input)
    if (audio_input_from_filename == False) and audio_input_from_array == True:                                           
        x = audio_input

    retriever = MIR(x, 44100)

    frame_size = 4096

    hop_size = 1024
 
    segments = [len(frame) / 44100 for frame in retriever.FrameGenerator()]

    output = []
    for segment in segments:                                           
        sample = int(segment*44100) 
        output.append(x[:sample*lenght]) #extend duration of segment

    output = choice(output)                                           

    if save_file == True:                                          
        baseName = os.path.splitext(audio_input)[0].split('/')[-1]                                                                       
        outputFilename = 'samples'+'/'+baseName+'_sample'+'.wav'                                                       
        # write_file(outputFilename, 44100, output)
        MonoWriter(filename = outputFilename)(output)
        print(("File generated: %s"%outputFilename))
    if save_file == False:
        return output

# ext_filter = ['.wav', '.mp3'] #valid audio files
ext_filter = ['.wav'] #valid audio files
Usage = "./DoSegmentation.py [FILES_DIR]"
def main():
    """
      Output: outputFilename = 'samples'+'/'+baseName+'_sample'+'.wav'
    """
    if len(sys.argv) < 2:
        print(("\nBad amount of input arguments\n", Usage, "\n"))
        sys.exit(1)

    # segment_length = 6
    segment_length = 10
    # try:
    files_dir = sys.argv[1] 

    if not os.path.exists(files_dir):                         
        raise IOError("Must download sounds")

    for subdir, dirs, files in os.walk(files_dir):
        for f in files:
            # print(f)
            if not os.path.splitext(f)[1] in ext_filter:
                continue
            audio_input = subdir+'/'+f
            do_segmentation( audio_input, audio_input_from_filename = True, audio_input_from_array = False, sec_len = segment_length, save_file = True)
            # do_segmentation( audio_input)                            

# def main():
#     if len(sys.argv) < 2:
#         print(("\nBad amount of input arguments\n", Usage, "\n"))
#         sys.exit(1)


#     try:
#         files_dir = sys.argv[1] 

#         if not os.path.exists(files_dir):                         
#             raise IOError("Must download sounds")

#         for subdir, dirs, files in os.walk(files_dir):
#             for f in files:
#                 audio_input = subdir+'/'+f
#                 do_segmentation( audio_input)                            

#     except Exception:
#         sys.exit(1)

if __name__ == '__main__': 
    main()
