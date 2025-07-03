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
from .sonification.Sonification import hfc_onsets
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  

ext_filter = ['.mp3','.ogg','.wav','.wma', '.amr'] # check if all extensions are supported by the library
#ext_filter = ['.mp3','.ogg','.undefined','.wav','.wma','.mid', '.amr'] # .mid is not an audio file, why is the reason to have .undefined extension?

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
                'metadata.duration',
                'lowlevel.spectral_flux',
                'lowlevel.zero_crossing_rate',
                'rhythm.onsets_hfc',
                'rhythm.onsets_by_polar_distance',
                'highlevel.chords_sequence',
                'highlevel.danceability'
                ]

def process_file(inputSoundFile, tag_dir, input_filename):
    descriptors_dir = (tag_dir+'/'+'descriptores')

    if not os.path.exists(descriptors_dir):                         
           os.makedirs(descriptors_dir)                                
           print("Creando directorio para archivos .json")

    json_output = descriptors_dir + '/' + os.path.splitext(input_filename)[0] + ".json"
                              
    if os.path.exists(json_output) is True:                         
        pending_descriptions = []
        for i in descriptors:                                         
            if not i+'.mean' in list(json.load(open(json_output,'r')).keys()):
                pending_descriptions.append(i)
        if pending_descriptions == []:                                         
            raise IOError(".json already saved")
    if os.path.exists(json_output) is False:                         
           pass

    try:
        pending_descriptions
    except Exception as e:
        pending_descriptions = []

    input_signal, sampleRate = read(inputSoundFile)

    input_signal = mono_stereo(input_signal)

    retrieve = MIR(input_signal, sampleRate)

    retrieve.signal = retrieve.IIR(retrieve.signal, 40, 'highpass')

    retrieve.mel_bands_global()

    onsets = []                                                                                                          
    for i in retrieve.FrameGenerator():                                                           
        retrieve.window()                                                    
        retrieve.Spectrum()                                                                                                       
        retrieve.Phase(retrieve.fft(retrieve.frame))          
        onsets.append(retrieve.detect_by_polar) 
        
    retrieve.onsets_by_polar_distance(onsets)

    # compute for all frames in our audio and add it to the pool
    pool = defaultdict(list)

    i = 0                                        
    if not sampleRate == 96000:                                                                 
        retrieve.n_bands = 40                                                    
    else:                                                      
        retrieve_n_bands = 31 #we can't use 40 bands when fs is vinyl type, 31 is the limit  
        
    pcps = [] 

    for share in retrieve.spectrum_share():
    
        #loudness
        namespace = 'loudness'
        desc_name = namespace + '.level'
        retrieve.Loudness()
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            pool[desc_name].append(retrieve.loudness)

        #low level
        namespace = 'lowlevel'

        desc_name = namespace + '.spectral_centroid'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            c = retrieve.centroid()
            pool[desc_name].append(c)

        desc_name = namespace + '.spectral_contrast'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            eighth_contrast = retrieve.contrast()
            pool[desc_name].append(eighth_contrast[0])
            pool[namespace + '.spectral_valleys'].append(eighth_contrast[1])

        desc_name = namespace + '.mfcc'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            retrieve.MFCC_seq()
            pool[desc_name].append(retrieve.mfcc_seq)
            pool[desc_name+'_bands'].append(retrieve.mel_bands)

        desc_name = namespace + '.hfc'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            pool[desc_name].append(retrieve.hfc())

        # dissonance
        retrieve.spectral_peaks()
        pcps.append(hpcp(retrieve,12))
        retrieve.fundamental_frequency()
        retrieve.harmonic_peaks()
        desc_name = namespace + '.dissonance'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            pool[desc_name].append(retrieve.dissonance())

        retrieve.Envelope()
        retrieve.AttackTime()
        desc_name = 'sfx.logattacktime'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
            pool[desc_name].append(retrieve.attack_time)

        #inharmonicity
        desc_name = 'sfx.inharmonicity'
        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions): 
            try:
                pool[desc_name].append(retrieve.inharmonicity())
            except IndexError:
                pool[desc_name].append(1.0) #we say its 1.0 simply because there's only one harmonic peak, meaning other peaks easily depart

        if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions): 
            pass
        else: 
            i += 1
            print ("Processing Frame " + str(i))            

    #bpm
    namespace = 'rhythm'
    desc_name = namespace + '.bpm'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        retrieve.onsets_strength()
        retrieve.bpm()
        pool[desc_name].append(retrieve.tempo)
        pool['rhythm.bpm_ticks'].append(retrieve.ticks / retrieve.M)

    #zero crossing rate
    namespace = 'lowlevel'
    desc_name = namespace + '.zero_crossing_rate'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        zcr = retrieve.zcr()
        pool[desc_name].append(zcr)

    #spectral flux
    namespace = 'lowlevel'
    desc_name = namespace + '.spectral_flux'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        try:
            retrieve.mel_dbs
            pool[desc_name].append(np.mean(retrieve.flux(retrieve.mel_dbs)))
        except Exception as e:
            retrieve.mel_bands_global()
            pool[desc_name].append(np.mean(retrieve.flux(retrieve.mel_dbs)))

    #onsets by polar distance
    namespace = 'rhythm'
    desc_name = namespace + '.onsets_by_polar_distance'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        pool[desc_name].append(onsets_indexes)
            
    #onsets by high frequency content
    namespace = 'rhythm'
    desc_name = namespace + '.onsets_hfc'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        pool[desc_name].append(np.mean(hfc_onsets(retrieve.signal) / retrieve.fs))            

    #danceability
    namespace = 'highlevel'
    desc_name = namespace + '.danceability'
    if tag_dir == 'electronic':                                
        meany = np.loadtxt('means/y1.txt') #one more time
    if tag_dir == 'pop':                                
        meany = np.loadtxt('means/y2.txt') #you should be dancing
    if tag_dir == 'rock':                                
        meany = np.loadtxt('means/y3.txt') #sultans of swing
    if tag_dir == 'mapuche':                                
        meany = np.loadtxt('means/y4.txt') #mapuche dance
    if tag_dir == 'reggaeton':                                
        meany = np.loadtxt('means/y5.txt') #despacito
    if tag_dir != 'reggaeton' and tag_dir != 'electronic' and tag_dir != 'pop' and tag_dir != 'rock' and tag_dir != 'mapuche':                                
        meany = random.choice(['y1.txt','y2.txt','y3.txt','y4.txt','y5.txt'])
        meany = np.loadtxt('means/'+meany)
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        pool[desc_name].append(danceability(retrieve.signal[retrieve.onsets_indexes[0]:retrieve.onsets_indexes[0]+(retrieve.fs*8)], meany, retrieve.fs))        

    #chords sequence
    namespace = 'highlevel'
    desc_name = namespace + '.chords_sequence'
    retrieve.audio_signal_spectrum = [] #empty the buffers to make chord analysis
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions): 
        try:                                     
            pool[desc_name].append(chord_sequence(retrieve,pcps))
        except Exception as e:
            pool[desc_name].append('No chord') #drum loop, melody or tone 

    #duration
    namespace = 'metadata'
    desc_name = namespace + '.duration'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        pool[desc_name].append(retrieve.duration)

    print("Obtaining mean values")

    sorted_keys = ["lowlevel.dissonance", "lowlevel.mfcc_bands", "sfx.inharmonicity", "rhythm.bpm", "lowlevel.spectral_contrast", "lowlevel.spectral_centroid", "rhythm.bpm_ticks", "lowlevel.mfcc", "loudness.level", "metadata.duration","lowlevel.spectral_valleys", "sfx.logattacktime", "lowlevel.hfc", 'lowlevel.spectral_flux','lowlevel.zero_crossing_rate','rhythm.onsets_hfc','rhythm.onsets_flux','highlevel.danceability','highlevel.chords_sequence'] #sort the keys correctly to prevent bad scaling

    keys = []
    for k in sorted_keys:
        try:                                                
            keys.append((k+'.mean', np.mean(pool.get(k)).real))
        except Exception as e:    
            keys.append((k+'.mean', pool.get(k)))              
    pool_stats = OrderedDict(keys)

    print (pool_stats)

    if not os.path.exists(json_output):    
        with open(json_output, 'w') as outfile:       
            json.dump(pool_stats, outfile) #write to file 
    else:
        with open(json_output) as outfile:     
            data = json.load(outfile) 
            data.update(pool_stats)
        with open(json_output, 'w') as outfile:
            outfile.write(json.dumps(pool_stats, indent = 2)) 

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
