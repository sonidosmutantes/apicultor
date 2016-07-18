#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from subprocess import call

import json
from essentia import *
from essentia.standard import *

files_dir = "data"
ext_filter = ['.mp3','.ogg','.ogg','.wav'] #valid audio files
#TODO: check sample rate

# descriptors of interest
descriptors = [ 
                'lowlevel.spectral_centroid.mean',
                #'lowlevel.spectral_contrast.mean',
                #'lowlevel.dissonance.mean',
                #'lowlevel.hfc.mean',
                #'lowlevel.mfcc.mean',
                #'sfx.logattacktime.mean',
                #'sfx.inharmonicity.mean'
                ]

def process_file(inputSoundFile, frameSize = 1024, hopSize = 512):
    #TODO check stereo vs mono file
    # load our audio into an array
    audio = MonoLoader(filename = inputSoundFile)()
    sampleRate = 44100 #FIXME
        
    #method alias for extractors
    centroid = SpectralCentroidTime()
    levelExtractor = LevelExtractor()
    mfcc = MFCC()    
    hfc = HFC()
    dissonance = Dissonance()

    #more aliases (helper functions)
#    w = Windowing() #default windows
    w_hann = Windowing(type = 'hann')
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(sampleRate = sampleRate, orderBy='frequency')

    # compute for all frames in our audio and add it to the pool
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize, hopSize):
        frame_windowed = w_hann(frame)
        frame_spectrum = spectrum(frame_windowed)
    
        #low level
        c = centroid( frame_spectrum )
        pool.add('lowlevel.spectral_centroid', c)

        mfcc_melbands, mfcc_coeffs = mfcc( frame_spectrum )
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.mfcc_bands', mfcc_melbands)

        h = hfc( frame_spectrum )
        pool.add('lowlevel.hfc', h)

        # dissonance
        (frame_frequencies, frame_magnitudes) = spectral_peaks(frame_spectrum)
        frame_dissonance = dissonance(frame_frequencies, frame_magnitudes)
        pool.add( 'lowlevel.' + 'dissonance', frame_dissonance)
        
        # t frame
        l = levelExtractor(frame)
        pool.add('loudness.level',l)
          


    # Pool stats (mean, var)
    #aggrPool = PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)
    aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

    # write result to file
    # json_output = os.path.splitext(inputSoundFile)[0]+"-new.json"
    # YamlOutput(filename = json_output, format = 'json')(aggrPool)

    data = dict()
    #for dn in pool.descriptorNames(): data[dn] = pool[dn].tolist()
    for dn in aggrPool.descriptorNames():
        try:
            data[dn] = str( aggrPool[dn][0] )
        except:
            data[dn] = str( aggrPool[dn] )
    #print data
    json_output = os.path.splitext(inputSoundFile)[0]+".json"
    with open(json_output, 'w') as outfile:
         json.dump(data, outfile) #write to file

    print(json_output)
#()    

for subdir, dirs, files in os.walk(files_dir):
    for f in files:
        if os.path.splitext(f)[1] in ext_filter:
            audio_input = subdir+'/'+f
            print( audio_input )
            process_file( audio_input )
#

#process_file('data/982.ogg')
