#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from subprocess import call

import json
from essentia import *
from essentia.standard import *

<<<<<<< HEAD
#files_dir = "data"
files_dir = "samples"
=======
files_dir = "data/bajo"
>>>>>>> c5ad6939e7667e31997ed9625c849ec7bfd3bce1
ext_filter = ['.mp3','.ogg','.ogg','.wav'] #archivos de sonido válidos

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

def process_file(inputSoundFile, frameSize = 1024, hopSize = 512):
    input_signal = MonoLoader(filename = inputSoundFile)()
    sampleRate = 44100
    
    #filter direct current noise
    offset_filter = DCRemoval()    
    #method alias for extractors
    centroid = SpectralCentroidTime()
    contrast = SpectralContrast(frameSize = frameSize+1)
    levelExtractor = LevelExtractor()
    mfcc = MFCC()    
    hfc = HFC()
    dissonance = Dissonance()
    bpm = RhythmExtractor2013()
    timelength = Duration()
    logat = LogAttackTime()
    harmonic_peaks = HarmonicPeaks()                                   
    f0_est = PitchYin()    
    inharmonicity = Inharmonicity()

    #++++helper functions++++
    envelope = Envelope()
#    w = Windowing() #default windows
    w_hann = Windowing(type = 'hann')
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(sampleRate = sampleRate, orderBy='frequency')
    audio = offset_filter(input_signal)

    # compute for all frames in our audio and add it to the pool
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize, hopSize):
        frame_windowed = w_hann(frame)
        frame_spectrum = spectrum(frame_windowed)
    
        #low level
        namespace = 'lowlevel'

        desc_name = namespace + '.spectral_centroid'
        if desc_name in descriptors:
            c = centroid( frame_spectrum )
            pool.add(desc_name, c)

        desc_name = namespace + '.spectral_contrast'
        if desc_name in descriptors:
            contrasts, valleys = contrast(frame_spectrum)
            pool.add(desc_name, contrasts)
            pool.add('lowlevel.spectral_valleys', valleys)

        desc_name = namespace + '.mfcc'
        if desc_name in descriptors:
            mfcc_melbands, mfcc_coeffs = mfcc( frame_spectrum )
            pool.add(desc_name, mfcc_coeffs)
            pool.add('lowlevel.mfcc_bands', mfcc_melbands)

        desc_name = namespace + '.hfc'
        if desc_name in descriptors:
            h = hfc( frame_spectrum )
            pool.add(desc_name, h)

        # dissonance
        desc_name = namespace + '.dissonance'
        if desc_name in descriptors:
            frame_frequencies, frame_magnitudes = spectral_peaks(frame_spectrum)
            frame_dissonance = dissonance(frame_frequencies, frame_magnitudes)
            pool.add( desc_name, frame_dissonance)

        
        # t frame
        namespace = 'loudness'
        desc_name = namespace + '.level'
        if desc_name in descriptors:
            l = levelExtractor(frame)
            pool.add(desc_name,l)

        #logattacktime
        desc_name = 'sfx.logattacktime'
        if desc_name in descriptors:
            frame_envelope = envelope(frame)
            attacktime = logat(frame_envelope)
            pool.add(desc_name, attacktime)

	#inharmonicity
        desc_name = 'sfx.inharmonicity'
        if desc_name in descriptors:
            pitch = f0_est(frame_windowed)
            frame_frequencies, frame_magnitudes = spectral_peaks(frame_spectrum)                             
            harmonic_frequencies, harmonic_magnitudes = harmonic_peaks(frame_frequencies[1:], frame_magnitudes[1:], pitch[0])                         
            inharmonic = inharmonicity(harmonic_frequencies, harmonic_magnitudes)      
            pool.add(desc_name, inharmonic)                       

        #bpm
        namespace = 'rhythm'
        desc_name = namespace + '.bpm'
        if desc_name in descriptors:
            beatsperminute = bpm(audio)[0]
            pool.add(desc_name, beatsperminute)

        #duration
        namespace = 'metadata'
        desc_name = namespace + '.duration'
        if desc_name in descriptors:
            duration = timelength(audio)
            pool.add(desc_name, duration)
    #end of frame computation


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
    print data
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
