#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import json
import numpy as np
import scipy
from essentia import *
from essentia.standard import *
from smst.utils.audio import write_wav

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

    #++++more helpers++++
    onsets_location = Onsets()
    detect_by_hfc = OnsetDetection(method = 'hfc')
    fft = FFT()

    cartesian_to_polar = CartesianToPolar()
    audio_f0 = PitchYinFFT()(spectrum(w_hann(audio)))[0]

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
            beatsperminute, ticks = bpm(audio)[0], bpm(audio)[1]
            pool.add(desc_name, beatsperminute)
            pool.add('rhythm.bpm_ticks', ticks)

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

    descriptors_dir = (tag_dir+'/'+'descriptores')

    if not os.path.exists(descriptors_dir):                         
           os.makedirs(descriptors_dir)                                
           print "Creando directorio para archivos .json"

    json_output = descriptors_dir + '/' + os.path.splitext(input_filename)[0] + ".json"
    with open(json_output, 'w') as outfile:
         json.dump(data, outfile) #write to file

    print(json_output)

    #sound recording with tempo marker

    tempo_dir = (tag_dir+'/'+'tempo')

    if not os.path.exists(tempo_dir):                         
           os.makedirs(tempo_dir)                                
           print "Creando directorio para marcado de tempo"

    mark_ticks = AudioOnsetsMarker(onsets=ticks, type='beep')                                             
                                              
    signal_bpm = mark_ticks(audio)
    print ("Saving File with tempo beeps") 
    output = write_wav(signal_bpm, 44100, tempo_dir + '/' + os.path.splitext(input_filename)[0] + 'tempo.wav')

    #spectral centroid of sound recording

    centroid_dir = (tag_dir+'/'+'centroid')

    if not os.path.exists(centroid_dir):                         
           os.makedirs(centroid_dir)                                
           print "Creando directorio para paso de banda de centroide"

    band_pass_filter = BandPass(cutoffFrequency = float(data['lowlevel.spectral_centroid.mean']))
    print ("Filtering Signal")
    signal_centroid = band_pass_filter(audio)
    output = write_wav(signal_centroid, 44100, centroid_dir + '/' + os.path.splitext(input_filename)[0] + 'centroid.wav')

    #mfccs of sound recording

    mfcc_dir = (tag_dir+'/'+'mfcc')

    if not os.path.exists(mfcc_dir):                         
           os.makedirs(mfcc_dir)                                
           print "Creando directorio para paso de banda de mfcc"

    melband_pass_filter = BandPass(cutoffFrequency = abs(float(data['lowlevel.mfcc.mean'])))
    print ("Filtering Signal according to Mel bands mean")
    signal_mfcc = melband_pass_filter(audio)
    output = write_wav(signal_mfcc, 44100, mfcc_dir + '/' + os.path.splitext(input_filename)[0] + 'mfcc.wav')

    #inharmonicity of sound recording

    inharmonicity_dir = (tag_dir+'/'+'inharmonicity')

    if not os.path.exists(inharmonicity_dir):                         
           os.makedirs(inharmonicity_dir)                                
           print "Creando directorio para filtrado por inarmonia"

    inharmonicity_pass_filter = AllPass(cutoffFrequency = audio_f0
 * (1 + float(data['sfx.inharmonicity.mean'])), bandwidth = 55)
    print ("Filtering Signal according to Inharmonicity mean")
    signal_inharmonicity = inharmonicity_pass_filter(audio)
    output = write_wav(signal_inharmonicity, 44100, inharmonicity_dir + '/' + os.path.splitext(input_filename)[0] + 'inharmonicity.wav')

    #dissonance of sound recording

    dissonance_dir = (tag_dir+'/'+'dissonance')

    if not os.path.exists(dissonance_dir):                         
           os.makedirs(dissonance_dir)                                
           print "Creando directorio para escucha de disonancia"

    dissonant_f = audio_f0 + 2.27*(pow(audio_f0, 0.4777))/(1 + float(data['lowlevel.dissonance.mean']))
    dissonance_pass_filter = BandPass(cutoffFrequency = dissonant_f, bandwidth = 55)
    print ("Filtering Signal according to Dissonance mean")
    signal_dissonance = dissonance_pass_filter(audio)
    output = write_wav(signal_dissonance, 44100, dissonance_dir + '/' + os.path.splitext(input_filename)[0] + 'dissonance.wav')

    #loudness sound recording

    loudness_dir = (tag_dir+'/'+'loudness')

    if not os.path.exists(loudness_dir):                         
           os.makedirs(loudness_dir)                                
           print "Creando directorio para escucha de loudness"                                          

    audio_loud = 10*np.log10(float(data['loudness.level.mean'])) * audio 
    maximum = np.max(np.abs(audio_loud))/-1 
    loud_sound = np.true_divide(audio_loud, maximum) 
    print ("Saving Loud Sound")
    output = write_wav(loud_sound, 44100, loudness_dir + '/' + os.path.splitext(input_filename)[0] + 'loudness.wav') 

    #sound recording based on valleys

    valleys_dir = (tag_dir+'/'+'valleys')

    if not os.path.exists(valleys_dir):                         
           os.makedirs(valleys_dir)                                
           print "Creando directorio para escucha de contraste espectral basado en valle espectral"                                           

    from smst.models.harmonic import from_audio
    from smst.models.sine import to_audio
    freq, mag, phase = from_audio(audio, 44100, scipy.hanning(1023), 2048, hopSize, -70, 20, audio_f0, audio_f0*2, 10, abs(float(data['lowlevel.spectral_contrast.mean'])))   
    contrast_enhancement = ((1.6 * mag) - (0.6*float(data['lowlevel.spectral_valleys.mean']))) 
    contrast = to_audio(freq, mag + contrast_enhancement, phase, frameSize, hopSize, 44100)  
    print ("Saving recording Contrast")
    output = write_wav(contrast, 44100, valleys_dir + '/' + os.path.splitext(input_filename)[0] + 'contrast.wav') 

    #sound recording with hfc marker

    hfc_dir = (tag_dir+'/'+'hfc')

    if not os.path.exists(hfc_dir):                         
           os.makedirs(hfc_dir)                                
           print "Creando directorio para marcado de contenido de frecuencia alta" 

    ctp_mag, ctp_phase = cartesian_to_polar(fft(w_hann(audio)))   
    marker = scipy.cos((2*scipy.pi*1000/44100)*scipy.arange(44100*(1.0/44100.0)))                                             

    audio[audio == audio[int(np.array(detect_by_hfc(ctp_mag, ctp_phase)).astype(int))]  ] = audio[audio == audio[int(np.array(detect_by_hfc(ctp_mag, ctp_phase)).astype(int))]  ] + marker  
    print ("Saving File with hfc bursts") 
    output = write_wav(audio, 44100, hfc_dir + '/' + os.path.splitext(input_filename)[0] + 'hfc.wav') 

    #sound recording with attack marker

    attack_dir = (tag_dir+'/'+'attack')

    if not os.path.exists(attack_dir):                         
           os.makedirs(attack_dir)                                
           print "Creando directorio para marcado de ataque" 

    at_s = 10**(float(data['sfx.logattacktime.mean']))  
    marker = scipy.cos((2*scipy.pi*1000/44100)*scipy.arange(44100*(1.0/44100.0)))                                             

    audio = offset_filter(input_signal)
    audio[audio == audio[(at_s)*44100]] = audio[audio == audio[(at_s)*44100]] + marker  
    print ("Saving File with attack marker") 
    output = write_wav(audio, 44100, attack_dir + '/' + os.path.splitext(input_filename)[0] + 'attack.wav') 

#()    

Usage = "./run_MIR_analysis.py [FILES_DIR]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        files_dir = sys.argv[1] 

    	if not os.path.exists(files_dir):                         
		raise IOError("Must download sounds")

	for subdir, dirs, files in os.walk(files_dir):
	    for f in files:
		    tag_dir = subdir
		    input_filename = f
		    audio_input = subdir+'/'+f
		    print( audio_input )
		    process_file( audio_input )                           

    except Exception, e:
        print(e)
        exit(1)

