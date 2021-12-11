#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
from scipy.io import wavfile

def fixed_segmentation(filename, options):
    """
        Segmenta con  duración fija
        Solo soporta wav files como input
        Si el archivo tiene una duración menor a la segmentación requerida, se graba
        en su duración original y nombre
    """

    if not '.wav' in filename:
        raise Exception("fixed_segmentation only process wav files")

    outputPath = options['outputPath']
    fixed_dur = options['duration']

    try:
        sr, wavsignal = wavfile.read(filename)

        durSamples = int( fixed_dur*sr )
        n_samples = len(wavsignal)

        baseName = os.path.splitext(filename)[0].split('/')[-1]

        if durSamples > n_samples:
            print("El archivo tiene una duración inferior a la segmentación requerida")
            print("Se graba el archivo en su duración original")
            outputFilename = outputPath+'/'+baseName+'.wav'
            wavfile.write(outputFilename,sr,np.array(wavsignal, dtype='int16'))
            print("File generated: %s"%outputFilename)
            return

        segments = int( np.ceil( n_samples/durSamples ) )
        pos = 0
        for i in range(segments):
            signalOut = wavsignal[pos:pos+durSamples]
            pos += durSamples
            outputFilename = outputPath+'/'+baseName+'_sample'+str(i)+'.wav'
            wavfile.write(outputFilename,sr,np.array(signalOut, dtype='int16'))
            print("File generated: %s"%outputFilename)
    except Exception as e:
        #TODO: add standard logging output
        print("Error: %s"%e)

Usage = "./fixed_segmentation.py [FILES_DIR] [JSON_DIR] [MAX_SEGMENT_DURATION]"
def main():
    if len(sys.argv) < 4:
        print("\nBad amount of input arguments\n\t", Usage, "\n")
        print(Usage)
        sys.exit(1)

    files_dir = sys.argv[1]
    output_path = sys.argv[2]
    max_segment_duration = sys.argv[3] # in seconds: 3, 5, 10, etc

    ext_filter = ['.wav'] # valid audio files
    options = dict()
    options['outputPath'] = output_path
    options['duration'] = int(max_segment_duration)

    for subdir, dirs, files in os.walk(files_dir):
        for f in files:
            if not os.path.splitext(f)[1] in ext_filter:
                        continue
            audio_input = subdir+'/'+f
            print( "Processing %s"%audio_input )
            fixed_segmentation( audio_input, options)

if __name__ == '__main__': 
    main()
