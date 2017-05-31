#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import sys
import json
import numpy as np
import logging
from .utils.algorithms import *
from soundfile import read
from collections import defaultdict, OrderedDict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  

ext_filter = ['.mp3','.ogg','.undefined','.wav','.wma','.mid', '.amr']

# descriptores de interés
descriptors = [ 
                'lowlevel.spectral_centroid',
                'lowlevel.spectral_contrast',
                'lowlevel.dissonance',
                'lowlevel.hfc',
                'lowlevel.mfcc',
                'loudness.level',
                'sfx.logattacktime',  
                'sfx.inharmonicity', 
                'rhythm.bpm',
                'metadata.duration'
                ]

def process_file(inputSoundFile, tag_dir, input_filename):
    descriptors_dir = (tag_dir+'/'+'descriptores')

    if not os.path.exists(descriptors_dir):                         
           os.makedirs(descriptors_dir)                                
           print("Creando directorio para archivos .json")

    json_output = descriptors_dir + '/' + os.path.splitext(input_filename)[0] + ".json"
                              
    if os.path.exists(json_output) is True:                         
           raise IOError(".json already saved")
    if os.path.exists(json_output) is False:                         
           pass

    input_signal, sampleRate = read(inputSoundFile)

    input_signal = mono_stereo(input_signal)

    retrieve = MIR(input_signal, sampleRate)

    retrieve.signal = retrieve.IIR(retrieve.signal, 40, 'highpass')

    # compute for all frames in our audio and add it to the pool
    pool = defaultdict(list)
    i = 0

    for frame in retrieve.FrameGenerator():
        retrieve.window()
        retrieve.Spectrum()
    
        #loudness
        namespace = 'loudness'
        desc_name = namespace + '.level'
        retrieve.Loudness()
        if desc_name in descriptors:
            pool[desc_name].append(retrieve.loudness)

        #low level
        namespace = 'lowlevel'

        desc_name = namespace + '.spectral_centroid'
        if desc_name in descriptors:
            c = retrieve.centroid()
            pool[desc_name].append(c)

        desc_name = namespace + '.spectral_contrast'
        if desc_name in descriptors:
            eighth_contrast = retrieve.contrast()
            pool[desc_name].append(eighth_contrast[0])
            pool[namespace + '.spectral_valleys'].append(eighth_contrast[1])

        desc_name = namespace + '.mfcc'
        if desc_name in descriptors:
            retrieve.MFCC_seq()
            pool[desc_name].append(retrieve.mfcc_seq)
            pool[desc_name+'_bands'].append(retrieve.mel_bands)

        desc_name = namespace + '.hfc'
        if desc_name in descriptors:
            pool[desc_name].append(retrieve.hfc())

        # dissonance
        retrieve.spectral_peaks()
        retrieve.fundamental_frequency()
        retrieve.harmonic_peaks()
        desc_name = namespace + '.dissonance'
        if desc_name in descriptors:
            pool[desc_name].append(retrieve.dissonance())

        retrieve.Envelope()
        retrieve.AttackTime()
        desc_name = 'sfx.logattacktime'
        if desc_name in descriptors:
            pool[desc_name].append(retrieve.attack_time)

        #inharmonicity
        desc_name = 'sfx.inharmonicity'
        if desc_name in descriptors: 
            pool[desc_name].append(retrieve.inharmonicity())

        i += 1
        print ("Processing Frame " + str(i))

    #bpm
    namespace = 'rhythm'
    desc_name = namespace + '.bpm'
    if desc_name in descriptors:
        retrieve.bpm()
        pool[desc_name].append(retrieve.tempo)
        pool['rhythm.bpm_ticks'].append(retrieve.ticks)

    #duration
    namespace = 'metadata'
    desc_name = namespace + '.duration'
    if desc_name in descriptors:
        pool[desc_name].append(retrieve.duration)

    #end of frame computation

    print("Obtaining mean values")

    sorted_keys = ["lowlevel.dissonance", "lowlevel.mfcc_bands", "sfx.inharmonicity", "rhythm.bpm", "lowlevel.spectral_contrast", "lowlevel.spectral_centroid", "rhythm.bpm_ticks", "lowlevel.mfcc", "loudness.level", "metadata.duration","lowlevel.spectral_valleys", "sfx.logattacktime", "lowlevel.hfc"] #sort the keys correctly to prevent bad scaling

    pool_stats = OrderedDict((k+'.mean', np.mean(pool.get(k)).real) for k in sorted_keys)
    print (pool_stats)

    with open(json_output, 'w') as outfile:
         json.dump(pool_stats, outfile) #write to file

    print(json_output) 
#()    

Usage = "./run_MIR_analysis.py [FILES_DIR]"
def main():  
    if len(sys.argv) < 2:
        print("\nBad amount of input arguments\n\t", Usage, "\n")
        print("Example:\n\t./run_MIR_analysis.py data\n\t./run_MIR_analysis.py samples\n")
        sys.exit(1)


    try:
        files_dir = sys.argv[1] 

        if not os.path.exists(files_dir):                         
            raise IOError("Must download sounds")

        error_count = 0
        for subdir, dirs, files in os.walk(files_dir):
            for f in files:
                if not os.path.splitext(f)[1] in ext_filter:
                    continue
                tag_dir = subdir
                input_filename = f
                audio_input = subdir+'/'+f
                try:
                    print(( "\n*** Processing %s\n"%audio_input ))
                    process_file( audio_input, tag_dir, input_filename )
                except Exception as e:
                    print(logger.exception(e))
                    error_count += 1 
                    continue					                		                         
        print(("Errors: %i"%error_count))
        sys.exit( -error_count )
    except Exception as e:
        logger.exception(e)
        exit(1)

if __name__ == '__main__': 
    main()
