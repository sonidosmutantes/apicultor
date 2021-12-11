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
from smst.utils.synth import *
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

bands = np.arange(5,5550,367)


def process_file(inputSoundFile, tag_dir, input_filename, genre):
    descriptors_dir = (tag_dir+'/'+'descriptores')

    if not os.path.exists(descriptors_dir):                         
           os.makedirs(descriptors_dir)                                
           print("Creando directorio para archivos .json")

    json_output = descriptors_dir + '/' + input_filename + "_band0.json"
    print(json_output)
                              
    if os.path.exists(json_output) is True:                         
        raise IOError('File exists')
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
        onsets.append(retrieve.detect_by_polar()) 
        
    #retrieve.onsets_by_polar_distance(onsets)

    # compute for all frames in our audio and add it to the pool
    pool = defaultdict(list)

    #i = 0                                        
    #if not sampleRate == 96000:                                                                 
    #    retrieve.n_bands = 40                                                    
    #else:                                                      
    #    retrieve_n_bands = 31 #we can't use 40 bands when fs is vinyl type, 31 is the limit  
    
    #genre = sys.argv[2]
        
    pcps = [] 
    first = True
    j = 0
    retrieve.onsets_by_flux()
    if genre == 'reggaeton':
        print(np.loadtxt('/home/mc/Descargas/means/reggaeton.txt'))
        print(np.loadtxt('/root/Descargas/reggaeton'))
        y = np.loadtxt('/home/mc/Descargas/means/reggaeton.txt')
        w = np.loadtxt('/root/Descargas/reggaeton')
    elif genre == 'pop':
        y = np.loadtxt('/home/mc/Descargas/means/pop.txt')
        w = np.loadtxt('/root/Descargas/pop')
    elif genre == 'romantico':
        y = np.loadtxt('/home/mc/Descargas/means/romantico.txt')
        w = np.loadtxt('/root/Descargas/romantico')
    for share in retrieve.audio_signal_spectrum:
        retrieve.magnitude_spectrum = share
        retrieve.spectral_peaks()
        mags = retrieve.magnitudes
        freq = retrieve.frequencies   
        band_split = int(mags.size/16)
        lower = 0
        retrieve.frame = retrieve.frames_onset[j]
        retrieve.Envelope()
        retrieve.AttackTime()   
        zcr = retrieve.zcr()     
        eighth_contrast = retrieve.contrast()
        retrieve.MFCC_seq()
        Danceability = danceability(retrieve.frame, np.array([w]), y, retrieve.fs)
        for i in range(16):
            try:
                retrieve.M = len(mags)
                retrieve.magnitudes = mags[lower:lower+band_split]
                retrieve.frequencies = freq[lower:lower+band_split]
                retrieve.phase = retrieve.phase_signal[i][lower:lower+band_split]
                if first is True:
                    pool['lowlevel.complexity'].append([])
                pool['lowlevel.complexity'][i].append(retrieve.magnitudes.size)
                moments = central_moments(retrieve)
                rolloff = roll_off(retrieve.magnitude_spectrum,retrieve.fs)
                if first is True:
                    pool['lowlevel.rolloff'].append([])
                pool['lowlevel.rolloff'][i].append(rolloff)
                kurtosis = dist_shape(moments)[2]
                if first is True:
                    pool['lowlevel.kurtosis'].append([])
                pool['lowlevel.kurtosis'][i].append(kurtosis)
                namespace = 'lowlevel'
                desc_name = namespace + '.spectral_centroid'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    c = retrieve.centroid()
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(c)
                desc_name = namespace + '.hfc'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(retrieve.hfc())
                retrieve.fundamental_frequency()
                retrieve.harmonic_peaks()
                if retrieve.harmonic_magnitudes.size is 1:
                    retrieve.harmonic_magnitudes = np.append(retrieve.harmonic_magnitudes,retrieve.harmonic_magnitudes)
                if retrieve.harmonic_frequencies.size is 1:
                    retrieve.harmonic_frequencies = np.append(retrieve.harmonic_frequencies/2,retrieve.harmonic_frequencies)
                desc_name = namespace + '.dissonance'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(retrieve.dissonance())
                desc_name = 'sfx.inharmonicity'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions): 
                    try:
                        if first is True:
                            pool[desc_name].append([])
                        pool[desc_name][i].append(retrieve.inharmonicity())
                    except Exception as e:
                        try:
                            pool[desc_name][i]
                        except Exception as e:
                            pass
                        pool[desc_name][i].append(1)
                desc_name = 'sfx.logattacktime'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                     if first is True:
                         pool[desc_name].append([])
                     pool[desc_name][i].append(retrieve.attack_time)
                namespace = 'loudness'
                desc_name = namespace + '.level'
                retrieve.Loudness()
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(retrieve.loudness)
                namespace = 'lowlevel'
                desc_name = namespace + '.zero_crossing_rate'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(zcr)
                namespace = 'highlevel'
                desc_name = namespace + '.danceability'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(Danceability) 
                namespace='lowlevel'
                desc_name = namespace + '.spectral_contrast'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:  
                        pool[desc_name].append([])
                    pool[desc_name][i].append(np.mean(eighth_contrast[0]))
                    if first is True:
                        pool[namespace+'.spectral_valleys'].append([])
                    pool[namespace + '.spectral_valleys'][i].append(np.mean(eighth_contrast[1]))
                desc_name = namespace + '.mfcc'
                if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
                    if first is True:
                        pool[desc_name].append([])
                    pool[desc_name][i].append(np.mean(retrieve.mfcc_seq))
                    if first is True:
                        pool[desc_name+'_bands'].append([])
                    pool[desc_name+'_bands'][i].append(np.mean(retrieve.mel_bands))   
                lower += band_split
                print ("Processing Frame " + str(i))
            except Exception as e:
                print(e,desc_name)
                continue
        first = False
        j += 1

    KEYS = []
    for band in range(16):
        KEYS.append([])
        for key in pool.keys():
            KEYS[band].append((key,np.mean(pool[key][band])))
        odict = OrderedDict(KEYS[band])
        with open(descriptors_dir+'/'+input_filename+'_band'+str(band)+'.json','w') as f:
            json.dump(odict, f)

    if True:
        return      

    #bpm
    namespace = 'rhythm'
    desc_name = namespace + '.bpm'
    if (desc_name in descriptors and os.path.exists(json_output) is False) or (pending_descriptions != [] and desc_name in pending_descriptions):
        retrieve.onsets_strength()
        retrieve.bpm()
        pool[desc_name].append(retrieve.tempo)
        pool['rhythm.bpm_ticks'].append(retrieve.ticks / retrieve.M)

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
        genre = sys.argv[2]

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
                    process_file( audio_input, tag_dir, input_filename, genre)
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
