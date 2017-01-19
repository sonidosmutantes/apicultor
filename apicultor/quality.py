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

#hiss removal (a noise reduction algorithm working on signal samples to reduce its hissings)
def hiss_removal(audio):
    pend = len(audio)-(2048+255)     
    noise_fft = FFT(size = 2048)(Windowing(size = 1023, type = 'hann')(audio[:2048]))
    noise_power = np.log10(np.abs(noise_fft + 2 ** -16))
    noise_floor = np.exp(2.0 * noise_power.mean())                                                     
    energy = lambda mag: np.sum((10 ** (mag / 20)) ** 2)                                     
    mn = Spectrum(size = 2048)(audio[:2048])
    e_n = energy(mn)   
    pin = 0                                 
    pinpin = 0                
    output = np.zeros(len(audio)) 
    while pin < pend:            
        p1 = int(pin)                        
        ft = FFT(size = 2048)(Windowing(size = 1023, type = 'hann')(audio[p1:p1+2048]))              
        m = Spectrum(size = 2048)(audio[p1:p1+2048])
        e_m = energy(m)
        SNR = 10 * np.log10(e_m / e_n)
        power_spectral_density = np.abs(ft) ** 2 
        if SNR > 0:                
            pass                                
        else:                    
            if np.any(power_spectral_density < noise_floor):                                    
                ft[power_spectral_density < noise_floor] *= 0.12589254117941673 #gate everything below the noise floor
                print ("Reducing noise floor, this is taking some time") 
            else:
                pass                 
        output[pinpin:pinpin+2048] += Windowing(type = 'hann', size = 1023)(IFFT(size = 2048)(ft))
        pinpin += 255                                               
        pin += 255
    amp = audio.max()
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
