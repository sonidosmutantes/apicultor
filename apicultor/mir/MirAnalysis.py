#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import numpy as np
import scipy
from essentia import *
from essentia.standard import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  

#ext_filter = ['.mp3','.ogg','.undefined','.wav','.wma','.mid', '.amr', '.au]
ext_filter = ['.mp3','.ogg', '.wav', '.au']

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

sampleRate = 44100
#sampleRate = 22050
print("WARNING: sampleRate hardcoded %i"%sampleRate)

def process_file(inputSoundFile, frameSize = 1024, hopSize = 512):
    descriptors_dir = (tag_dir+'/'+'descriptores')

    if not os.path.exists(descriptors_dir):                         
           os.makedirs(descriptors_dir)                                
           print "Creando directorio para archivos .json"

    global input_filename
    json_output = descriptors_dir + '/' + os.path.splitext(input_filename)[0] + ".json"
                              
	# Disable json overwrite
#    if os.path.exists(json_output) is True:                         
#           raise IOError(".json already saved")
#    if os.path.exists(json_output) is False:                         
#           pass

    input_signal = MonoLoader(filename = inputSoundFile)()
#    sampleRate = 44100
    global sampleRate

    #filter direct current noise
    offset_filter = DCRemoval()    
    #method alias for extractors
    centroid = SpectralCentroidTime(sampleRate = sampleRate)
    contrast = SpectralContrast(frameSize = frameSize+1)
    levelExtractor = LevelExtractor()
    mfcc = MFCC(inputSize = 513)    
    hfc = HFC(sampleRate = sampleRate)
    dissonance = Dissonance()
    bpm = RhythmExtractor2013()
    timelength = Duration()
    logat = LogAttackTime(sampleRate = sampleRate)
    harmonic_peaks = HarmonicPeaks()                                   
    f0_est = PitchYin()    
    inharmonicity = Inharmonicity()

    #++++helper functions++++
    envelope = Envelope(sampleRate = sampleRate)
#    w = Windowing() #default windows
    w_hann = Windowing(type = 'hann')
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(sampleRate = sampleRate, orderBy='frequency')
    audio = offset_filter(input_signal)

    # compute for all frames in our audio and add it to the pool
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize, hopSize, startFromZero = True, lastFrameToEndOfFile=True):
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

        #logattacktime #TODO Latest version of Essentia currently returns three values of LogAttackTime, we should decide to use exponentials or use latest Essentia's output in s
        #desc_name = 'sfx.logattacktime'
        #if desc_name in descriptors:
        #    frame_envelope = envelope(frame)
        #    attacktime = logat(frame_envelope)
        #    pool.add(desc_name, attacktime)

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
            beatsperminute, ticks = bpm(audio)[0], bpm(audio)[1]
            pool.add(desc_name, beatsperminute)
            # pool.add('rhythm.bpm_ticks', ticks)
            pool.add('rhythm.bpm', beatsperminute)

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
    # FIXME: por ej el duration no tiene sentido calcularle el 'mean'

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

    with open(json_output, 'w') as outfile:
         json.dump(data, outfile) #write to file

    print(json_output) 
#()    

Usage = "./run_MIR_analysis.py [FILES_DIR]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n\t", Usage, "\n"
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
                    print( "\n*** Processing %s\n"%audio_input )
                    process_file( audio_input )
                except Exception, e:
                    print logger.exception(e)
                    error_count += 1 
                    continue					                		                         
        print("Errors: %i"%error_count)
        sys.exit( -error_count )
    except Exception, e:
        logger.exception(e)
        exit(1)
