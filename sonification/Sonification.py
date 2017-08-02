#!/usr/bin/python
# -*- coding: UTF-8 -*-

from utils.dj import hfc_onsets
import os
import sys
import json
import numpy as np
import scipy
from essentia import *
from essentia.standard import *
import logging
from smst.models.harmonic import from_audio
from smst.models.sine import to_audio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  

def mir_sonification(inputSoundFile, data):
    input_signal = MonoLoader(filename = inputSoundFile)()
    sampleRate = 44100
    
    #filter direct current noise
    offset_filter = DCRemoval()                                     
    f0_est = PitchYin()    
    #++++helper functions++++
    w_hann = Windowing(type = 'hann')
    spectrum = Spectrum()
    audio = offset_filter(input_signal)

    #++++more helpers++++
    onsets_location = Onsets()

    audio_f0 = PitchYinFFT()(spectrum(w_hann(audio)))[0]

    descriptors_dir = (tag_dir+'/'+'descriptores') #descriptors directory of tag

    #sound recording with tempo marker

    tempo_dir = (tag_dir+'/'+'tempo')

    if not os.path.exists(tempo_dir):                         
           os.makedirs(tempo_dir)                                
           print "Creando directorio para marcado de tempo"

    ticks = RhythmExtractor2013()(audio)[1] 
    mark_ticks = AudioOnsetsMarker(onsets=ticks, type='beep')                                             
                                              
    signal_bpm = mark_ticks(audio)
    print ("Saving File with tempo beeps") 
    output = MonoWriter(filename = tempo_dir + '/' + os.path.splitext(input_filename)[0] + 'tempo.ogg', format = 'ogg')(signal_bpm)

    #spectral centroid of sound recording

    centroid_dir = (tag_dir+'/'+'centroid')

    if not os.path.exists(centroid_dir):                         
           os.makedirs(centroid_dir)                                
           print "Creando directorio para paso de banda de centroide"

    band_pass_filter = BandPass(cutoffFrequency = float(data['lowlevel.spectral_centroid.mean']))
    print ("Filtering Signal")
    signal_centroid = band_pass_filter(audio)
    output = MonoWriter(filename = centroid_dir + '/' + os.path.splitext(input_filename)[0] + 'centroid.ogg', format = 'ogg')(signal_centroid)

    #mfccs of sound recording

    mfcc_dir = (tag_dir+'/'+'mfcc')

    if not os.path.exists(mfcc_dir):                         
           os.makedirs(mfcc_dir)                                
           print "Creando directorio para paso de banda de mfcc"

    melband_pass_filter = BandPass(cutoffFrequency = abs(float(data['lowlevel.mfcc.mean'])))
    print ("Filtering Signal according to Mel bands mean")
    signal_mfcc = melband_pass_filter(audio)
    output = MonoWriter(filename = mfcc_dir + '/' + os.path.splitext(input_filename)[0] + 'mfcc.ogg', format = 'ogg')(signal_mfcc)

    #inharmonicity of sound recording

    inharmonicity_dir = (tag_dir+'/'+'inharmonicity')

    if not os.path.exists(inharmonicity_dir):                         
           os.makedirs(inharmonicity_dir)                                
           print "Creando directorio para filtrado por inarmonia"

    inharmonicity_pass_filter = AllPass(cutoffFrequency = audio_f0
 * (1 + float(data['sfx.inharmonicity.mean'])), bandwidth = 55)
    print ("Filtering Signal according to Inharmonicity mean")
    signal_inharmonicity = inharmonicity_pass_filter(audio)
    output = MonoWriter(filename = inharmonicity_dir + '/' + os.path.splitext(input_filename)[0] + 'inharmonicity.ogg', format = 'ogg')(signal_inharmonicity)

    #dissonance of sound recording

    dissonance_dir = (tag_dir+'/'+'dissonance')

    if not os.path.exists(dissonance_dir):                         
           os.makedirs(dissonance_dir)                                
           print "Creando directorio para escucha de disonancia"

    dissonant_f = audio_f0 + 2.27*(pow(audio_f0, 0.4777))/(1 + float(data['lowlevel.dissonance.mean']))
    dissonance_pass_filter = BandPass(cutoffFrequency = dissonant_f, bandwidth = 55)
    print ("Filtering Signal according to Dissonance mean")
    signal_dissonance = dissonance_pass_filter(audio)
    output = MonoWriter(filename = dissonance_dir + '/' + os.path.splitext(input_filename)[0] + 'dissonance.ogg', format = 'ogg')(signal_dissonance)

    #loudness sound recording

    loudness_dir = (tag_dir+'/'+'loudness')

    if not os.path.exists(loudness_dir):                         
           os.makedirs(loudness_dir)                                
           print "Creando directorio para escucha de loudness"                                          

    audio_loud = 10*np.log10(float(data['loudness.level.mean'])) * audio 
    maximum = np.max(np.abs(audio_loud))/-1 
    loud_sound = np.true_divide(audio_loud, maximum) 
    print ("Saving Loud Sound")
    output = MonoWriter(filename = loudness_dir + '/' + os.path.splitext(input_filename)[0] + 'loudness.ogg', format = 'ogg')(loud_sound) 

    #sound recording based on valleys

    valleys_dir = (tag_dir+'/'+'valleys')

    if not os.path.exists(valleys_dir):                         
           os.makedirs(valleys_dir)                                
           print "Creando directorio para escucha de contraste espectral basado en valle espectral"                                           

    freq, mag, phase = from_audio(audio, sampleRate, scipy.hanning(1023), 2048, 512, -70, 20, audio_f0, audio_f0*2, 10, abs(float(data['lowlevel.spectral_contrast.mean'])))   
    contrast_enhancement = ((1.3 * mag) - (0.3*float(data['lowlevel.spectral_valleys.mean']))) 
    contrast = to_audio(freq, mag + contrast_enhancement, phase, 1025, 512, sampleRate)  
    print ("Saving recording Contrast")
    output = MonoWriter(filename = valleys_dir + '/' + os.path.splitext(input_filename)[0] + 'contrast.ogg', format = 'ogg')(array(contrast)) 

    #sound recording with hfc marker

    hfc_dir = (tag_dir+'/'+'hfc')

    if not os.path.exists(hfc_dir):                         
           os.makedirs(hfc_dir)                                
           print "Creando directorio para marcado de contenido de frecuencia alta" 

    hfcs = hfc_onsets(audio)
    print ("Saving File with hfc bursts") 
    output = MonoWriter(filename = hfc_dir + '/' + os.path.splitext(input_filename)[0] + 'hfc.ogg', format = 'ogg')(AudioOnsetsMarker(onsets=hfcs, type='beep')(audio)) 

    #sound recording with attack marker

    attack_dir = (tag_dir+'/'+'attack')

    if not os.path.exists(attack_dir):                         
           os.makedirs(attack_dir)                                
           print "Creando directorio para marcado de ataque" 

    at_s = 10**(float(data['sfx.logattacktime.mean']))                                              

    audio = offset_filter(input_signal)
    audio[audio == audio[(at_s)*sampleRate]] = audio[audio == audio[(at_s)*sampleRate]] + marker  
    print ("Saving File with attack marker") 
    output = MonoWriter(filename = attack_dir + '/' + os.path.splitext(input_filename)[0] + 'attack.ogg', format = 'ogg')(audio)     

Usage = "./Sonification.py [FILES_DIR]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nNeed tag dir\n", Usage, "\n"
        sys.exit(1)


    try:
        files_dir = sys.argv[1]
        descriptors_dir = files_dir+'/descriptores/' 

    	if not os.path.exists(files_dir):                         
		raise IOError("Must download sounds")

	for subdir, dirs, files in os.walk(files_dir):
	    for f in files:
		    tag_dir = subdir
		    input_filename = f
		    audio_input = subdir+'/'+f
		    print( audio_input )
		    with open(descriptors_dir+f.split('.')[0]+'.json') as mir_data:
		    	data = json.load(mir_data)
		    mir_sonification(audio_input, data)                           

    except Exception, e:
        logger.exception(e)
        exit(1)



