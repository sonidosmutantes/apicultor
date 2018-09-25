#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from ..constraints.dynamic_range import dyn_constraint_satis
from ..utils.algorithms import *
from ..sonification.Sonification import normalize, write_file
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import lfilter, fftconvolve, firwin
from soundfile import read
import os
import sys
import logging

#TODO: *Remove wows, clippings, clicks and pops

#you should comment what you've already processed (avoid over-processing) 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

energy = lambda mag: np.sum((10 ** (mag / 20)) ** 2)  

rel = lambda at: at * 10 #calculate release time

to_coef = lambda at, sr: np.exp((np.log(9)*-1) / (sr * at)) #convert attack and release time to coefficients

#hiss removal (a noise reduction algorithm working on signal samples to reduce its hissings)

def hiss_removal(audio):
    pend = len(audio)-(2205+551)
    song = sonify(audio, 44100) 
    song.FrameGenerator().__next__()
    song.window() 
    song.Spectrum()
    noise_fft = song.fft(song.windowed_x)[:song.H+1]
    noise_power = np.log10(np.abs(noise_fft + 2 ** -16))
    noise_floor = np.exp(2.0 * noise_power.mean())                                    
    mn = song.magnitude_spectrum
    e_n = energy(mn)   
    pin = 0                
    output = np.zeros(len(audio))
    hold_time = 0
    ca = 0
    cr = 0
    amp = audio.max()
    while pin < pend:
        selection = pin+1024
        song.frame = audio[pin:selection] 
        song.window()               
        song.Spectrum()
        e_m = energy(song.magnitude_spectrum)
        SNR = 10 * np.log10(e_m / e_n)
        ft = song.fft(song.windowed_x)[:song.H+1]
        power_spectral_density = np.abs(ft) ** 2
        song.Envelope()
        song.AttackTime()
        rel_time = rel(song.attack_time)
        rel_coef = to_coef(rel_time, 44100)
        at_coef = to_coef(song.attack_time, 44100)
        ca = ca + song.attack_time
        cr = cr + rel_time 
        if SNR > 0:                
            np.add.at(output, range(pin, selection), audio[pin:selection])                                
        else:                    
            if np.any(power_spectral_density < noise_floor):                                    
                gc = dyn_constraint_satis(ft, [power_spectral_density, noise_floor], 0.12589254117941673) 
                if ca > hold_time:
                    gc = np.complex64([at_coef * gc[i- 1] + (1 - at_coef) * x if x > gc[i- 1] else x for i,x in enumerate(gc)])
                if ca <= hold_time:
                    gc = np.complex64([gc[i- 1] for i,x in enumerate(gc)])
                if cr > hold_time:
                    gc = np.complex64([rel_coef * gc[i- 1] + (1 - rel_coef) * x if x <= gc[i- 1] else x for i,x in enumerate(gc)])
                if cr <= hold_time:
                    gc = np.complex64([gc[i- 1] for i,x in enumerate(gc)])
                print ("Reducing noise floor, this is taking some time")
                song.Phase(song.fft(song.windowed_x))
                song.phase = song.phase[:song.magnitude_spectrum.size]
                ft *= gc
                song.magnitude_spectrum = np.sqrt(pow(ft.real,2) + pow(ft.imag,2))
                np.add.at(output, range(pin, selection), song.ISTFT(song.magnitude_spectrum))
            else:
                np.add.at(output, range(pin, selection), audio[pin:selection])                                              
        pin = pin + song.H
        hold_time += selection/44100
    hissless = amp * output / output.max() #amplify to normal level                                                 
    return np.float32(hissless) 

#optimizers and biquad_filter taken from Linear Audio Lib
def z_from_f(f,fs):
    out = []
    for x in f:
        if x == np.inf:
            out.append(-1.) 
        else:
            out.append(((fs/np.pi)-x)/((fs/np.pi)+x))
    return out

def Fz_at_f(Poles,Zeros,f,fs,norm = 0):
    omega = 2*np.pi*f/fs
    ans = 1.
    for z in Zeros:
        ans = ans*(np.exp(omega*1j)-z_from_f([z],fs))
    for p in Poles:
        ans = ans/(np.exp(omega*1j)-z_from_f([p],fs))
    if norm:
        ans = ans/max(abs(ans))
    return ans

def z_coeff(Poles,Zeros,fs,g,fg,fo = 'none'):
    if fg == np.inf:
        fg = fs/2
    if fo == 'none':
        beta = 1.0
    else:
        beta = f_warp(fo,fs)/fo
    a = np.poly(z_from_f(beta*np.array(Poles),fs))
    b = np.poly(z_from_f(beta*np.array(Zeros),fs))
    gain = 10.**(g/20.)/abs(Fz_at_f(beta*np.array(Poles),beta*np.array(Zeros),fg,fs))
    
    return (a,b*gain)

def biquad_filter(xin,z_coeff):
    a = z_coeff[0]
    b = z_coeff[1]
    xout = np.zeros(len(xin))
    xout[0] = b[0]*xin[0]
    xout[1] = b[0]*xin[1] + b[1]*xin[0] - a[1]*xout[0]
    
    for j in range(2,len(xin)):
        xout[j] = b[0]*xin[j]+b[1]*xin[j-1]+b[2]*xin[j-2]-a[1]*xout[j-1]-a[2]*xout[j-2]

    return xout

RIAA = [[50.048724,2122.0659],[500.48724,np.inf]]

Usage = "./quality.py [DATA_PATH] [HTRF_SIGNAL_PATH]"

def main():
    if len(sys.argv) < 3:
        print("\nBad amount of input arguments\n", Usage, "\n")
        sys.exit(1)


    try:
        DATA_PATH = sys.argv[1]

        if not os.path.exists(DATA_PATH):                         
            raise IOError("Must download sounds")

        for subdir, dirs, files in os.walk(DATA_PATH):                  
            for f in files:                                           
                print(( "Rewriting without hissing in %s"%f ))          
                audio,fs = read(DATA_PATH+'/'+f) 
                audio = mono_stereo(audio)              
                hissless = hiss_removal(audio) #remove hiss             
                print(( "Rewriting without crosstalk in %s"%f ))        
                hrtf = read('hrtf.wav')[0] #load the hrtf wav file                                          
                b = firwin(2, [0.05, 0.95], width=0.05, pass_zero=False)            
                convolved = fftconvolve(hrtf, b[np.newaxis, :], mode='valid') 
                left = convolved[:int(convolved.shape[0]/2), :] 
                right = convolved[int(convolved.shape[0]/2):, :]    
                h_sig_L = lfilter(left.flatten(), 1., audio)             
                h_sig_R = lfilter(right.flatten(), 1., audio)            
                del hissless  
                result = np.float32([h_sig_L, h_sig_R]).T               
                neg_angle = result[:,(1,0)]         
                panned = result + neg_angle
                normalized = normalize(panned)                                         
                audio = mono_stereo(normalized)          
                print(( "Rewriting without aliasing in %s"%f ))         
                song = sonify(audio, 44100)                     
                audio = song.IIR(audio, 44100/2, 'lowpass') #anti-aliasing filtering: erase frequencies higher than the sample rate being used
                print(( "Rewriting without DC in %s"%f ))                                  
                audio = song.IIR(audio, 40, 'highpass') #remove direct current on audio signal                       
                print(( "Rewriting with Equal Loudness contour in %s"%f ))                                
                audio = song.EqualLoudness(audio) #Equal-Loudness Contour  
                normalized_riaa = normalize(audio)                   
                del audio                           
                print(( "Rewriting with Hum removal applied in %s"%f ))
                song.signal = np.float32(normalized_riaa)                
                without_hum = song.BandReject(np.float32(normalized_riaa), 50, 16) #remove undesired 50 hz hum     
                del normalized_riaa                                     
                print(( "Rewriting with subsonic rumble removal applied in %s"%f ))                          
                song.signal = without_hum                               
                without_rumble = song.IIR(song.signal, 20, 'highpass') #remove subsonic rumble                  
                del without_hum                                         
                db_mag = 20 * np.log10(abs(without_rumble)) #calculate silence if present in audio signal                         
                print(( "Rewriting without silence in %s"%f ))
                silence_threshold = -130 #complete silence              
                loud_audio = np.delete(without_rumble, np.where(db_mag < silence_threshold))#remove it
                print(( "Rewriting with RIAA filter applied in %s"%f )) 
                abz = z_coeff(RIAA[0],RIAA[1],fs,0,10000) 
                riaa_filtered = biquad_filter(loud_audio, abz)  #riaa filter  
                riaa_filtered = normalize(riaa_filtered)               
                write_file(subdir+'/'+os.path.splitext(f)[0], fs, riaa_filtered)                                                              
                del without_rumble                       

    except Exception as e:
        logger.exception(e)
        exit(1)

if __name__ == '__main__': 
    main()
