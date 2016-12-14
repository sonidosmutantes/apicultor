from smst.utils.math import to_db_magnitudes
import numpy as np
from essentia.standard import *
import os
import sys

#TODO: *Remove wows, clippings, clicks & pops, rumble, hisses, process sound with crosstalk removal (sth like ambiophonics)

#you should comment what you've already processed (avoid over-processing)

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
		    print( "Rewriting without alising in %s"%f )
		    audio = MonoLoader(filename = DATA_PATH+'/'+f)()
		    audio = LowPass(cutoffFrequency = 44100/2)(audio) #anti-aliasing filtering: erase frequencies higher than the sample rate being used
		    print( "Rewriting without DC in %s"%f )
		    audio = DCRemoval()(audio) #remove direct current on audio signal
		    print( "Rewriting with RIAA filter applied in %s"%f )
		    riaa_filtered = biquad_filter(audio, abz)  #riaa filter 
		    maximum_normalizing = np.max(np.abs(riaa_filtered))/-1 
		    normalized_riaa = np.true_divide(riaa_filtered,maximum_normalizing) 
		    print( "Rewriting with Hum removal applied in %s"%f )
                    without_hum = BandReject(bandwidth = 16, cutoffFrequency=50)(np.float32(normalized_riaa)) #remove undesired 50 hz hum 
		    db_mag = to_db_magnitudes(without_hum) #calculate silence if present in audio signal
		    print( "Rewriting without silence in %s"%f )
		    silence_threshold = -130 #complete silence
		    loud_audio = np.delete(without_hum, np.where(db_mag <  silence_threshold))#remove it
		    os.remove(subdir+'/'+f)
		    MonoWriter(filename = subdir+'/'+os.path.splitext(f)[0]+'.ogg', format = 'ogg')(loud_audio)                         

    except Exception, e:
        print(e)
        exit(1)
