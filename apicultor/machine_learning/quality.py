#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import librosa
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

def width_interpolation(w_idx):
    w_interp = []
    for i in range(1):
        w_interp.append((w_idx[i]*( 1-((np.sin(2*np.pi*1/3)+1)/2.0) ) ) + ( w_idx[i-1]*((np.sin(2*np.pi*1/3)+1)/2.0)))
    return w_interp
    
w = np.load('width.npy')

w_inter = width_interpolation(w)    

def lstm_synth_predict(audio):
    stft = librosa.stft(audio,hop_length=1024,win_length=2048).T
    output = librosa.istft((stft*w_inter).T,hop_length=1024)
    return librosa.util.normalize(output)

def greatestCommonDivisor(x, y,epsilon): 
  if (x<y):  
      return greatestCommonDivisor(y,x,epsilon); 
  if (x==0): 
      return 0; 
  error = 2147483647 
  ratio = 2147483647 
  bpmDistance(x,y,error,ratio); 
  if (abs(error)<epsilon): 
      return y; 
  a = int(x+0.5); 
  b = int(y+0.5); 
  while (abs(error) > epsilon): 
    bpmDistance(a,b,error,ratio); 
    remainder = a%b; 
    a=b; 
    b=remainder; 
  return a; 

def bpmDistance(x,y, error, ratio): 
  ratio = x/y; 
  error = -1; 
  if (ratio < 1): 
    ratio = round(1./ratio); 
    error=(x*ratio-y)/min(y,Real(x*ratio))*100; 
  else: 
    ratio = round(ratio); 
    error = (x-y*ratio)/min(x,Real(y*ratio))*100; 
  return error, ratio 

def areEqual(a, b, tolerance): 
  error=0; 
  ratio=0; 
  bpmDistance(a,b,error,ratio); 
  return (abs(error)<tolerance) and (int(ratio)==1);

def HarmonicBpm():
  harmonicBpms = np.zeros(bpms.size);
  harmonicRatios = np.zeros(bpms.size);
  for i in range(bpms.size):
    ratio = _bpm/bpms[i];
    if (ratio < 1): ratio = 1.0/ratio;
    gcd = greatestCommonDivisor(_bpm, bpms[i], _tolerance);
    if (gcd > _threshold):
      harmonicBpms[i] = bpms[i]
      if (gcd < mingcd): mingcd = gcd;
    
  harmonicBpms = np.sort(harmonicBpms)
  i=0;
  prevBestBpm = -1;
  while i<harmonicBpms.size:
    prevBpm = harmonicBpms[i];
    while i < harmonicBpms.size:
      areEqual(prevBpm,harmonicBpms[i], _tolerance) 
      error=0
      r=0;
      bpmDistance(_bpm, harmonicBpms[i], error, r);
      error = abs(error);
      if (error < minError):
        bestBpm = harmonicBpms[i];
        minError = error;
    i += 1 
    if not areEqual(prevBestBpm, bestBpm, _tolerance): bestHarmonicBpms[bestBpm]
    else:
      e1=0, 
      e2=0, 
      r1=0, 
      r2=0;
      bpmDistance(_bpm, bestHarmonicBpms[bestHarmonicBpms.size-1], e1, r1);
      bpmDistance(_bpm, bestBpm, e2, r2);
      e1 = abs(e1);
      e2 = abs(e2);
      if (e1 > e2):
        bestHarmonicBpms[bestHarmonicBpms.size()-1] = bestBpm;
    prevBestBpm = bestBpm;
  return bestHarmonicBpms

def constantQ_transform(audio):
    pin = 0                
    output = np.zeros(audio.size)
    pend = audio.size
    while pin < pend:
        selection = pin+2048
        song.frame = audio[pin:selection] 
        constantQ = constantq.cqt(song.frame,sr=song.fs,hop_length=1024,n_bins=113)
        output[pin:selection] = constantq.icqt(constantQ,sr=song.fs,hop_length=1024)
        pin += 1024
    return output


def hiss_removal(audio):
    pend = len(audio)-(4410+1102)
    song = sonify(audio, 44100) 
    song.FrameGenerator().__next__()
    song.window() 
    song.Spectrum()
    noise_fft = song.fft(song.windowed_x,fft=False)[:song.H+1]
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
        selection = pin+2048
        song.frame = audio[pin:selection] 
        song.window()     
        song.M = 2048            
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


Usage = "./quality.py [DATA_PATH] [HTRF_SIGNAL_PATH]"

def main():
    if len(sys.argv) < 3:
        print("\nBad amount of input arguments\n", Usage, "\n")
        sys.exit(1)


    try:
        DATA_PATH = sys.argv[1]
        RIAA = [[50.048724,2122.0659],[500.48724,np.inf]]
        abz = z_coeff(RIAA[0],RIAA[1],44100,0,10000) 

        if not os.path.exists(DATA_PATH):                         
            raise IOError("Must download sounds")

        for subdir, dirs, files in os.walk(DATA_PATH):                  
            for f in files:                                           
                print(( "Rewriting without hissing in %s"%f ))          
                audio = read(DATA_PATH+'/'+f)[0] 
                audio = mono_stereo(audio)              
                #hissless = hiss_removal(audio) #remove hiss             
                #print(( "Rewriting without crosstalk in %s"%f ))        
                #hrtf = read(sys.argv[2])[0] #load the hrtf wav file                                          
                #b = firwin(2, [0.05, 0.95], width=0.05, pass_zero=False)  

                #convolved = fftconvolve(hrtf, b, mode='valid') 
                #convolved = np.vstack((convolved,convolved))
                #left = convolved[:int(convolved.shape[0]/2), :] 
                #right = convolved[int(convolved.shape[0]/2):, :]   
                #h_sig_L = lfilter(left.flatten(), 1., audio)             
                #h_sig_R = lfilter(right.flatten(), 1., audio)            
                #del hissless  
                #result = np.float32([h_sig_L, h_sig_R]).T               
                #neg_angle = result[:,(1,0)]         
                #panned = result + neg_angle
                #normalized = normalize(panned)  
                #del normalized                                          
                #audio = mono_stereo(audio)          
                print(( "Rewriting without aliasing in %s"%f ))         
                song = sonify(audio, 44100)                     
                audio = song.IIR(audio, 20000, 'lowpass') #anti-aliasing filtering: erase frequencies higher than the sample rate being used
                print(( "Rewriting without DC in %s"%f ))                                  
                audio = song.IIR(audio, 40, 'highpass') #remove direct current on audio signal                       
                print(( "Rewriting with Equal Loudness contour in %s"%f ))                                
                audio = song.EqualLoudness(audio) #Equal-Loudness Contour  
                print(( "Rewriting with RIAA filter applied in %s"%f )) 
                riaa_filtered = biquad_filter(audio, abz)  #riaa filter 
                del audio
                normalized_riaa = normalize(riaa_filtered)                   
                del riaa_filtered                           
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
                write_file(subdir+'/'+os.path.splitext(f)[0], 44100, loud_audio)                                                              
                del without_rumble                       

    except Exception as e:
        logger.exception(e)
        exit(1)

if __name__ == '__main__': 
    main()
