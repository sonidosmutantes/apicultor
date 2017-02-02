from smst.utils.math import to_db_magnitudes, from_db_magnitudes
from smst.models import stft
import numpy as np
from essentia.standard import *
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import resample, filtfilt, get_window
import os
import sys

#TODO: *Remove wows, clippings, clicks and pops

#you should comment what you've already processed (avoid over-processing)   

at = lambda samples: np.exp(LogAttackTime()(samples)) #calculate attack time

rel = lambda at: at * 10 #calculate release time

to_coef = lambda at, sr: np.exp((np.log(9)*-1) / (sr * at)) #convert attack and release time to coefficients

def dyn_constraint_satis(audio, variables, gain):
    assert ((type(variables) is list) and len(variables) == 2)
    audio[variables[0] < variables[1]] = gain  #noise gating: anything below a threshold is silenced     
    audio[variables[0] > variables[1]] = 1
    return audio

#hiss removal (a noise reduction algorithm working on signal samples to reduce its hissings)

def hiss_removal(audio):
    pend = len(audio)-(2048+255)     
    noise_fft = FFT(size = 2048)(Windowing(size = 1023, type = 'hann')(audio[:2048]))
    noise_power = np.log10(np.abs(noise_fft + 2 ** -16))
    noise_floor = np.exp(2.0 * noise_power.mean())                                                     
    energy = lambda mag: np.sum((10 ** (mag / 20)) ** 2)  
    dyn = lambda mag: np.sum((10 ** (mag / 20)) ** 2)                                     
    mn = Spectrum(size = 2048)(audio[:2048])
    e_n = energy(mn)   
    pin = 0                
    output = np.zeros(len(audio))
    hold_time = 0
    ca = 0
    cr = 0
    while pin < pend:
        selection = pin+2048
        frame = audio[pin:selection]                      
        ft = FFT(size = 2048)(Windowing(size = 1023, type = 'hann')(frame))              
        m = Spectrum(size = 2048)(frame)
        e_m = energy(m)
        SNR = 10 * np.log10(e_m / e_n)
        power_spectral_density = np.abs(ft) ** 2
        env = Envelope()(audio[pin:selection])
        attack_time = at(env)
        rel_time = rel(attack_time)
        rel_coef = to_coef(rel_time, 44100)
        at_coef = to_coef(attack_time, 44100)
        ca += attack_time
        cr += rel_time 
        if SNR > 0:                
            pass                                
        else:                    
            if np.any(power_spectral_density < noise_floor):                                    
                gc = dyn_constraint_satis(ft, [power_spectral_density, noise_floor], 0.12589254117941673) 
                if ca > hold_time:
                    gc = np.complex64([at_coef * gc[i- 1] + (1 - at_coef) * x if x > gc[i- 1] else x for i,x in enumerate(gc)])
                if ca <= hold_time:
                    gc = np.complex64([gc[i- 1] for i,x in enumerate(gc)])
                if cr > hold_time:
                    gc = np.complex64([at_coef * gc[i- 1] + (1 - at_coef) * x if x <= gc[i- 1] else x for i,x in enumerate(gc)])
                if cr <= hold_time:
                    gc = np.complex64([gc[i- 1] for i,x in enumerate(gc)])
                print ("Reducing noise floor, this is taking some time")
                ft = ft * gc
            else:
                pass                 
        output[pin:selection] += Windowing(type = 'hann', size = 1023)(IFFT(size = 2048)(ft))                                               
        pin += 255
        hold_time += selection/44100
    amp = audio.max()
    hissless = amp * output / output.max() #amplify to normal level                                                 
    return np.float32(hissless) 


Usage = "./quality.py [DATA_PATH]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        DATA_PATH = sys.argv[1]
        RIAA = [[50.048724,2122.0659],[500.48724,np.inf]]
        abz = z_coeff(RIAA[0],RIAA[1],44100,0,10000) 

    	if not os.path.exists(DATA_PATH):                         
		raise IOError("Must download sounds")

	for subdir, dirs, files in os.walk(DATA_PATH):
	    for f in files:
		    print( "Rewriting without hissing in %s"%f )
		    audio = MonoLoader(filename = DATA_PATH+'/'+f)()
		    hissless = hiss_removal(audio) #remove hiss
		    print( "Rewriting without crosstalk in %s"%f )
		    hrtf = AudioLoader(filename = 'H0e030a.wav')() #load the hrtf wav file
		    hrtf = hrtf[0]
		    h_sig_L = filtfilt(hrtf[:,0], 1., audio) 
		    h_sig_R = filtfilt(hrtf[:,1], 1., audio)
		    result = np.float32([h_sig_L, h_sig_R]).T
		    neg_angle = np.float32([h_sig_R, h_sig_L]).T
		    panned = result + neg_angle
		    maximum_normalizing = np.max(np.abs(panned))/-1 
		    normalized = np.true_divide(panned,maximum_normalizing) 
		    os.remove(subdir+'/'+f)
		    AudioWriter(filename = subdir+'/'+os.path.splitext(f)[0]+'.mp3', format = 'mp3')(normalized) #write a tmp file
		    audio = MonoLoader(filename = subdir+'/'+os.path.splitext(f)[0]+'.mp3')() #load it
		    os.remove(subdir+'/'+os.path.splitext(f)[0]+'.mp3') #we've loaded the audio data, so now the tmp file can be deleted
		    print( "Rewriting without aliasing in %s"%f )
		    audio = LowPass(cutoffFrequency = 44100/2)(audio) #anti-aliasing filtering: erase frequencies higher than the sample rate being used
		    print( "Rewriting without DC in %s"%f )
		    audio = DCRemoval()(audio) #remove direct current on audio signal
		    print( "Rewriting with Equal Loudness contour in %s"%f )
		    audio = EqualLoudness()(audio) #remove direct current on audio signal
		    print( "Rewriting with RIAA filter applied in %s"%f )
		    riaa_filtered = biquad_filter(audio, abz)  #riaa filter 
		    maximum_normalizing = np.max(np.abs(riaa_filtered))/-1 
		    normalized_riaa = np.true_divide(riaa_filtered,maximum_normalizing) 
		    print( "Rewriting with Hum removal applied in %s"%f )
                    without_hum = BandReject(bandwidth = 16, cutoffFrequency=50)(np.float32(normalized_riaa)) #remove undesired 50 hz hum 
		    print( "Rewriting with subsonic rumble removal applied in %s"%f )
                    without_rumble = HighPass(cutoffFrequency=20)(np.float32(without_hum)) #remove subsonic rumble 
		    db_mag = to_db_magnitudes(without_rumble) #calculate silence if present in audio signal
		    print( "Rewriting without silence in %s"%f )
		    silence_threshold = -130 #complete silence
		    loud_audio = np.delete(without_rumble, np.where(db_mag <  silence_threshold))#remove it
		    MonoWriter(filename = subdir+'/'+os.path.splitext(f)[0]+'.mp3', format = 'mp3')(loud_audio)                         

    except Exception, e:
        print(e)
        exit(1)
