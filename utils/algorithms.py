#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from scipy.signal import *
import numpy as np
from librosa import util
from librosa import magphase, clicks, time_to_frames
from librosa.onset import onset_detect
from librosa.beat import __beat_track_dp as dp
from librosa.beat import __last_beat as last_beat
from librosa.beat import __trim_beats as trim_beats
from scipy.fftpack import fft, ifft
from collections import Counter
from ..machine_learning.cache import memoize
from ..gradients.ascent import *
from ..gradients.descent import *
import random

p = lambda r, n: (r * np.sqrt(n-2)) / np.sqrt(1-(r**2))

def mono_stereo(input_signal): 
    input_signal = input_signal.astype(np.float32)   
    if len(input_signal.shape) != 1:               
        input_signal = input_signal.sum(axis = 1)/2
        return input_signal
    else:                  
        return input_signal

db2amp = lambda value: 0.5*pow(10, value/10) 

m_to_hz = lambda m: 700.0 * (np.exp(m/1127.01048) - 1.0)

hz_to_m = lambda hz: 1127.01048 * np.log(hz/700.0 + 1.0)

a_weight = lambda f: 1.25893 * pow(12200,2) * pow(f, 4) / ((pow(f, 2) + pow(20.6, 2)) * (pow(f,2) + pow(12200,2)) * np.sqrt(pow(f,2) + pow(107.7,2)) * np.sqrt(pow(f,2) + pow(737.9,2))) #a weighting of frequencies

bark_critical_bandwidth = lambda bark: 52548.0 / (pow(bark,2) - 52.56 * bark + 690.39) #compute a critical bandwidth with Bark frequencies

rms = lambda signal: np.sqrt(np.mean(np.power(signal,2))) 

def polar_to_cartesian(m, p): 
    real = m * np.cos(p) 
    imag = m * np.sin(p)  
    return real + (imag*1j) 

def plomp(frequency_difference):
    """Computes Plomp's consonance from difference between frequencies that might be dissonant"""
    if (frequency_difference < 0): #it's consonant
        return 1
    if (frequency_difference > 1.18): #it's consonant too
        return 1
    res = -6.58977878 * pow(frequency_difference, 5) + 28.58224226 * pow(frequency_difference, 4) + -47.36739986 * pow(frequency_difference, 3) + 35.70679761 * pow(frequency_difference, 2) + -10.36526344 * frequency_difference + 1.00026609 #try to solve for inbox frequency if it is consonant
    if (res < 0): #it's absolutely not consonant
        return 0
    if (res > 1):
        return 1 #it's absolutely consonant
    return res

def bark(f): 
    """Convert a frequency in Hz to a Bark frequency"""  
    bark = ((26.81*f)/(1960 + f)) - 0.53
    if (bark < 2):
        bark += 0.15 * (2-bark)
    if (bark > 20.1):
        bark += 0.22*(bark-20.1)
    return bark 

def bark_to_hz(bark):  
    """Convert a Bark frequency to a frequency in Hz"""                                               
    if (bark < 2):                                                    
        bark = (bark - 0.3) / 0.85
    if (bark > 20.1):
        bark = (bark - 4.422) / 1.22
    return 1960.0 * (bark + 0.53) / (26.28 - bark)

def consonance(f1, f2):
    """Compute the consonance between a pair of frequencies"""
    critical_bandwidth_f1 = bark_critical_bandwidth(bark(f1)) #get critical bandwidths
    critical_bandwidth_f2 = bark_critical_bandwidth(bark(f2))
    critical_bandwidth = min(critical_bandwidth_f1, critical_bandwidth_f2) #the least critical is the critical bandwidth 
    return plomp(abs(critical_bandwidth_f2 - critical_bandwidth_f1) / critical_bandwidth) #calculate Plomp's consonance

def morph(x, y, H, smoothing_factor, balance_factor):
    L = int(x.size / H)
    pin = 0
    ysong = sonify(y, 44100)
    xsong = sonify(x, 44100)
    xsong.N = 1024
    xsong.M = 512
    xsong.H = 256
    ysong = sonify(x, 44100)
    ysong.N = 1024
    ysong.M = 512
    ysong.H = 256
    selection = H
    z = np.zeros(x.size)
    for i in range(L):
        xsong.frame = x[pin:selection]
        ysong.frame = y[pin:selection]
        xsong.window()
        ysong.window()
        xsong.Spectrum()
        ysong.Spectrum()
        xmag = xsong.magnitude_spectrum
        ymag = ysong.magnitude_spectrum
        ymag_smoothed = resample(np.maximum(1e-10, ymag), round(ymag.size * smoothing_factor))
        ymag = resample(ymag_smoothed, xmag.size)
        output_mag = balance_factor * ymag + (1 - balance_factor) * xmag
        xsong.phase = np.angle(xsong.fft(xsong.windowed_x))[:257].ravel()
        invert = xsong.ISTFT(output_mag)
        z[pin:selection] += invert
        pin += H
        selection += H
    return z 

def hann(M):  
     _window = np.zeros(M)
     for i in range(M):                      
         _window[i] = 0.5 - 0.5 * np.cos((2.0*np.pi*i) / (M - 1.0))
     return _window 

def triang(M):        
    return np.array([2.0/M * (M/2.0 - abs(i - (M-1.) / 2.)) for i in range(M)])

def blackmanharris(M):
    a0 = 0.35875 
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168      
    fConst = 2.0 * np.pi / (M - 1) 
    return [a0 - a1 * np.cos(fConst * i) +a2 * np.cos(fConst * 2 * i) - a3 * np.cos(fConst * 3 * i) for i in range(M)] 

def synth_window(M):
    t = triang(M)
    b = blackmanharris(M)
    b /= sum(b)
    return t / b
    
class MIR:
    def __init__(self, audio, fs):
        """MIR class works as a descriptor for Music IR tasks. A set of algorithms used in APICultor are included.
        -param: audio: the input signal
        -fs: sampling rate
        -type: the type of window to use"""
        self.signal = (mono_stereo(audio) if len(audio) == 2 else audio)
        self.fs = fs
        self.N = 2048
        self.M = 1024
        self.H = 512
        self.n_bands = 40
        self.type = type
        self.nyquist = lambda hz: hz/(0.5*self.fs)
        self.to_db_magnitudes = lambda x: 20*np.log10(np.abs(x))
        self.duration = len(self.signal) / self.fs
        self.audio_signal_spectrum = []
        self.flux = lambda spectrum: np.array([np.sqrt(sum(pow(spectrum[i-1] - spectrum[i], 2))) for i in range(len(spectrum))])

    def FrameGenerator(self):
        """Creates frames of the input signal based on detected peak onsets, which allow to get information from pieces of audio with much information"""
        frames = int(len(self.signal) / self.H) 
        total_size = self.M
        for i in range(frames):
            self.frame = self.signal[i*(self.M-self.H):(i*(self.M-self.H) + self.M)]
            if not len(self.frame) == total_size:
                break 
            elif all(self.frame == 0):           
                print ("Frame values are all 0, discarding the frame")    
                continue  
            else:           
                yield self.frame

    def Envelope(self):
        _ga = np.exp(- 1.0 / (self.fs * 10))
        _gr = np.exp(- 1.0 / (self.fs * 1500))     
        _tmp = 0                           
        self.envelope = np.zeros(self.frame.size)
        for i in range(len(self.frame)):
            if _tmp < self.frame[i]:                           
                _tmp = (1.0 - _ga) * self.frame[i] + _ga * _tmp;
            else:                                               
                 _tmp = (1.0 - _gr) * self.frame[i] + _gr * _tmp;
            self.envelope[i] = _tmp
            if _tmp == round(float(str(_tmp)), 308): #find out if the number is denormal
                _tmp = 0

    def AttackTime(self): 
        start_at = 0.0;               
        cutoff_start_at = max(self.envelope) * 0.2 #initial start time
        stop_at = 0.0;      
        cutoff_stop_at = max(self.envelope) * 0.9 #initial stop time

        for i in range(len(self.envelope)):
            if self.envelope[i] >= cutoff_start_at:
                start_at = i
                break

        for i in range(len(self.envelope)):
            if self.envelope[i] >= cutoff_stop_at: 
                stop_at = i 
                break

        at_start = start_at / self.fs 
        at_stop = stop_at / self.fs 
        at = at_stop - at_start #we take the time in seconds directly
        if at < 0:
            self.attack_time = 0.006737946999085467 #this should be correct for non-negative output
        else:
            self.attack_time = at

    def IIR(self, array, cutoffHz, type):
        """Apply an Infinite Impulse Response filter to the input signal                                                                        
        -param: cutoffHz: cutoff frequency in Hz                        
        -type: the type of filter to use [highpass, bandpass]"""                                        
        if type == 'bandpass':                                          
            nyquist_low = self.nyquist(cutoffHz[0]) 
            nyquist_hi = self.nyquist(cutoffHz[1])                
            Wn = [nyquist_low, nyquist_hi] 
        else:                
            Wn = self.nyquist(cutoffHz)                   
        b,a = iirfilter(1, Wn, btype = type, ftype = 'butter')          
        output = lfilter(b,a,array) #use a time-freq compromised filter
        return output 

    def BandReject(self, array, cutoffHz, q):
        """Apply a 2nd order Infinite Impulse Response filter to the input signal  
        -param: cutoffHz: cutoff frequency in Hz                        
        -type: the type of filter to use [highpass, bandpass]"""                                        
        c = (np.tan(np.pi * q / self.fs)-1) / (np.tan(np.pi * q / self.fs)+1)
        d = -np.cos(2 * np.pi * cutoffHz / self.fs)
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] = (1.0-c)/2.0
        b[1] = d*(1.0-c)
        b[2] = (1.0-c)/2.0
        a[0] = 1.0
        a[1] = d*(1.0-c)
        a[2] = -c
        output = lfilter(b,a,array)
        return output 

    def AllPass(self, cutoffHz, q):
        """Apply an allpass filter to the input signal                                                                        
        -param: cutoffHz: cutoff frequency in Hz                        
        -q: bandwidth""" 
        w0 = 2. * np.pi * cutoffHz / self.fs
        alpha = np.sin(w0) / (2. * q)
        a = np.zeros(3)
        b = np.zeros(3)
        a[0] = (1.0 + alpha)
        a[1] = (-2.0 * np.cos(w0)) / a[0]
        a[2] = (1.0 - alpha) / a[0]
        b[0] = (1.0 - alpha) / a[0]
        b[1] = (-2.0 * np.cos(w0)) / a[0]
        b[2] = (1.0 + alpha) / a[0]
        output = lfilter(b, a, self.signal) 
        return output  

    def EqualLoudness(self, array):
        #Yulewalk filter                                                   
        By = np.zeros(11)                         
        Ay = np.zeros(11)          
                                   
        #Butterworth filter        
        Bb = np.zeros(3)           
        Ab = np.zeros(3)           
                                   
        By[0] =   0.05418656406430;
        By[1] =  -0.02911007808948;
        By[2] =  -0.00848709379851;
        By[3] =  -0.00851165645469;
        By[4] =  -0.00834990904936;
        By[5] =   0.02245293253339;
        By[6] =  -0.02596338512915;
        By[7] =   0.01624864962975;
        By[8] =  -0.00240879051584;                    
        By[9] =   0.00674613682247;
        By[10] = -0.00187763777362;

        Ay[0] =   1.00000000000000;                                        
        Ay[1] =  -3.47845948550071;               
        Ay[2] =   6.36317777566148;
        Ay[3] =  -8.54751527471874;
        Ay[4] =   9.47693607801280;
        Ay[5] =  -8.81498681370155;
        Ay[6] =   6.85401540936998;
        Ay[7] =  -4.39470996079559;
        Ay[8] =   2.19611684890774;
        Ay[9] =  -0.75104302451432;
        Ay[10] =  0.13149317958808;
                               
        Bb[0] =  0.98500175787242;
        Bb[1] = -1.97000351574484;
        Bb[2] =  0.98500175787242;
                               
        Ab[0] =  1.00000000000000;                     
        Ab[1] = -1.96977855582618;
        Ab[2] =  0.97022847566350;
                               
        output1 = lfilter(By, Ay, array)
        output = lfilter(Bb, Ab, output1)
        return output              
  
    def window(self):
        """Applies windowing to a frame"""  
        self.windowed_x = np.zeros(len(self.frame))
        w = hann(self.M)
        for i in range(self.H, self.M):
            self.windowed_x[i] = self.frame[i] * w[i]
        for i in range(0, self.H):
            self.windowed_x[i] = self.frame[i] * w[i]

        self.windowed_x *= (2 / sum(abs(self.windowed_x)))

    def fft(self, frame):
        N_min = int(self.N/32)     
                                                              
        n = np.arange(N_min)
        k = n[:, None]   
        M = np.exp(-2j * np.pi * n * k / N_min)
        transformed = np.dot(M, np.append(frame,np.zeros(self.M - (frame.size - self.M), dtype=np.complex128)).reshape((N_min, -1))).astype(np.complex128)
                                                       
        while transformed.shape[0] < transformed.size: 
            even = transformed[:, :int(transformed.shape[1] / 2)]
            odd = transformed[:, int(transformed.shape[1] / 2):]
            factor = np.exp(-1j * np.pi * np.arange(transformed.shape[0])
                            / transformed.shape[0])[:, None]   
            transformed = np.vstack([even + factor * odd,      
                           even - factor * odd])               
        return transformed.ravel()

    def ISTFT(self, signal, to_frame = True):
        """Computes the Inverse Short-Time Fourier Transform of a signal for sonification"""
        x_back = np.zeros(self.M, dtype = np.complex128)
        x_back[:self.H+1] = polar_to_cartesian(signal, self.phase)
        x_back[self.H+1:] = polar_to_cartesian(signal[-2:0:-1], self.phase[-2:0:-1])
        tmp = ifft(x_back, n = self.N).real[:self.M]
        return tmp

    def Phase(self, fft):
        """Computes phase spectrum from fft. The size is the same as the size of the magnitude spectrum""" 
        self.phase = np.arctan2(fft[:int(fft.size/4)+1].imag, fft[:int(fft.size/4)+1].real)

    def Spectrum(self):
        """Computes magnitude spectrum of windowed frame""" 
        self.spectrum = self.fft(self.windowed_x) 
        self.magnitude_spectrum = np.array([np.sqrt(pow(self.spectrum[i].real, 2) + pow(self.spectrum[i].imag, 2)) for i in range(self.H + 1)]).ravel() / self.H
        self.audio_signal_spectrum.append(self.magnitude_spectrum)

    def spectrum_share(self):
        """Give back all stored computations of spectrums of different frames. This is a generator""" 
        for spectrum in self.audio_signal_spectrum:        
            self.magnitude_spectrum = spectrum 
            yield self.magnitude_spectrum

    def onsets_by_flux(self):
        """Use this function to get only frames containing peak parts of the signal""" 
        self.onsets_indexes = np.where(self.flux(self.mel_dbs) > 80)[0]
        self.audio_signal_spectrum = np.array(self.audio_signal_spectrum)[self.onsets_indexes]

    def onsets_by_polar_distance(self,onsets):
        detection_rm = np.convolve(onsets/max(onsets), np.ones(2048), mode='valid')
        index = 0
        buffer = np.zeros(len(detection_rm))
        for i in range(len(detection_rm)):
            index = i+1 % len(buffer)
            if index == 0:
                buffer[:int(index)] = detection_rm[:int(index)] 
            else:
                if index == buffer.size:
                    index -= 1
            buffer[int(index)] = detection_rm[int(index)]
            threshold = np.median(buffer) + 0.00001 * np.mean(buffer)
        detection_rm = detection_rm[:np.array(self.audio_signal_spectrum).shape[0]]
        self.onsets_indexes = np.where(detection_rm>threshold)[0]
        self.audio_signal_spectrum = np.array(self.audio_signal_spectrum)[np.where(detection_rm[:len(self.audio_signal_spectrum)]>threshold)]
        
        
    def calculate_filter_freq(self):       
        band_freq = np.zeros(self.n_bands+2)
        low_mf = hz_to_m(0)
        hi_mf = hz_to_m(11000)
        mf_increment = (hi_mf - low_mf)/(self.n_bands+1)
        mel_f = 0
        for i in range(self.n_bands+2):
            band_freq[i] = m_to_hz(mel_f)
            mel_f += mf_increment
        return band_freq

    def MelFilter(self): 
        band_freq = self.calculate_filter_freq() 
        frequencyScale = (self.fs / 2.0) / (self.magnitude_spectrum.size - 1)                       
        self.filter_coef = np.zeros((self.n_bands, self.magnitude_spectrum.size))
        for i in range(self.n_bands):                          
            fstep1 = band_freq[i+1] - band_freq[i]   
            fstep2 = band_freq[i+2] - band_freq[i+1] 
            jbegin = int(round(band_freq[i] / frequencyScale + 0.5))           
            jend = int(round(band_freq[i+2] / frequencyScale + 0.5))
            if jend - jbegin <= 1:   
                raise ValueError 
            for j in range(jbegin, jend):
                bin_freq = j * frequencyScale                         
                if (bin_freq >= band_freq[i]) and (bin_freq < band_freq[i+1]):
                    self.filter_coef[i][j] = (bin_freq - band_freq[i]) / fstep1
                elif (bin_freq >= band_freq[i+1]) and (bin_freq < band_freq[i+2]):
                    self.filter_coef[i][j] = (band_freq[i+2] - bin_freq) / fstep2
        bands = np.zeros(self.n_bands)                             
        for i in range(self.n_bands):                                       
            jbegin = int(round(band_freq[i] / frequencyScale + 0.5))
            jend = int(round(band_freq[i+2] / frequencyScale + 0.5))
            for j in range(jbegin, jend):            
                bands[i] += pow(self.magnitude_spectrum[j],2) * self.filter_coef[i][j]
        return bands

    def DCT(self, array):                                                   
        self.table = np.zeros((self.n_bands,self.n_bands))
        scale = 1./np.sqrt(self.n_bands)
        scale1 = np.sqrt(2./self.n_bands)
        for i in range(self.n_bands):
            if i == 0:
                scale = scale
            else:
                scale = scale1
            freq_mul = (np.pi / self.n_bands * i)
            for j in range(self.n_bands):
                self.table[i][j] = scale * np.cos(freq_mul * (j + 0.5))
            dct = np.zeros(13)
            for i in range(13):
                dct[i] = 0
                for j in range(self.n_bands):
                    dct[i] += array[j] * self.table[i][j]
        return dct

    def MFCC_seq(self):
        """Computes Mel Frequency Cepstral Coefficients. It returns the Mel Bands using a Mel filter and a sequence of MFCCs""" 
        self.mel_bands = self.MelFilter()
        dbs = 2 * (10 * np.log10(self.mel_bands))
        self.mfcc_seq = self.DCT(dbs)

    def autocorrelation(self):
        self.N = (self.windowed_x[:,0].shape[0]+1) * 2
        corr = ifft(fft(self.windowed_x, n = self.N))                                   
        self.correlation = util.normalize(corr, norm = np.inf)
        subslice = [slice(None)] * np.array(self.correlation).ndim
        subslice[0] = slice(self.windowed_x.shape[0])                   
                                     
        self.correlation = np.array(self.correlation)[subslice]                                          
        if not np.iscomplexobj(self.correlation):               
            self.correlation = self.correlation.real
        return self.correlation

    def mel_bands_global(self):
        mel_bands = []   
        self.n_bands = 30                   
        for frame in self.FrameGenerator():                      
            try:
                self.window()                     
                self.Spectrum()                       
                mel_bands.append(self.MelFilter())
            except Exception as e:
                print((e))
                break
        self.mel_dbs = 2 * (10 * np.log10(abs(np.array(mel_bands))))

    def onsets_strength(self):   
        """Spectral Flux of a signal used for onset strength detection"""                                                              
        self.envelope = self.mel_dbs[:,1:] - self.mel_dbs[:,:-1]
        self.envelope = np.maximum(0.0, self.envelope)
        channels = [slice(None)] 
        self.envelope = util.sync(self.envelope, channels, np.mean, True, 0)
        pad_width = 1 + (2048//(2*self.H)) 
        self.envelope = np.pad(self.envelope, ([0, 0], [int(pad_width), 0]),mode='constant') 
        self.envelope = lfilter([1.,-1.], [1., -0.99], self.envelope, axis = -1) 
        self.envelope = self.envelope[:,:self.n_bands][0]  

    def bpm(self):   
        """Computes tempo of a signal in Beats Per Minute with its tempo onsets""" 
        self.onsets_strength()                                                             
        n = len(self.envelope) 
        win_length = np.asscalar(time_to_frames(8.0, self.fs, self.H))
        ac_window = hann(win_length) 
        self.envelope = np.pad(self.envelope, int(win_length // 2),mode='linear_ramp', end_values=[0, 0])
        frames = 1 + int((len(self.envelope) - win_length) / 1) 

        f = []                                                                            
        for i in range(win_length):     
            f.append(self.envelope[i:i+frames])
        f = np.array(f)[:,:n]              
        self.windowed_x = f * ac_window[:, np.newaxis]
        self.autocorrelation()

        tempogram = np.mean(self.correlation, axis = 1, keepdims = True)

        bin_frequencies = np.zeros(int(tempogram.shape[0]), dtype=np.float)

        bin_frequencies[0] = np.inf
        bin_frequencies[1:] = 60.0 * self.fs / (self.H * np.arange(1.0, tempogram.shape[0]))

        prior = np.exp(-0.5 * ((np.log2(bin_frequencies) - np.log2(80)) / bin_frequencies[1:].std())**2)
        max_indexes = np.argmax(bin_frequencies < 208)
        min_indexes = np.argmax(bin_frequencies < 80)

        prior[:max_indexes] = 0
        prior[min_indexes:] = 0
        p = prior.nonzero()

        best_period = np.argmax(tempogram[p] * prior[p][:, np.newaxis] * -1, axis=0)
        self.tempo = bin_frequencies[p][best_period]

        period = round(60.0 * (self.fs/self.H) / self.tempo[0])

        window = np.exp(-0.5 * (np.arange(-period, period+1)*32.0/period)**2)
        localscore = convolve(self.envelope/self.envelope.std(ddof=1), window, 'same')
        backlink, cumscore = dp(localscore, period, 100)
        self.ticks = [last_beat(cumscore)]

        while backlink[self.ticks[-1]] >= 0:
            self.ticks.append(backlink[self.ticks[-1]])

        self.ticks = np.array(self.ticks[::-1], dtype=int)

        self.ticks = trim_beats(localscore, self.ticks, False) * self.H
        if not len(self.ticks) >= 2:
            raise ValueError(("Only found one single onset, can't make sure if the beat is correct"))
        interv_value = self.ticks[1] - self.ticks[0] #these are optimal beat locations
        interval = 0
        self.ticks = []
        for i in range(int(self.signal.size/interv_value)):
            self.ticks.append(interval + interv_value)
            interval += interv_value #compute tempo frames locations based on the beat location value
        self.ticks = np.array(self.ticks) / self.fs
        return self.tempo, self.ticks

    def centroid(self):
        """Computes the Spectral Centroid from magnitude spectrum"""                                         
        a_pow_sum = 0                                                                                                       
        b_pow_sum = 0                                                                                                       
        for i in range(1, self.magnitude_spectrum.size):                                                                     
            a = self.magnitude_spectrum[i]                                                                                   
            b = self.magnitude_spectrum[i] - self.magnitude_spectrum[i-1]                                                     
            a_pow_sum += a * a                                                                                              
            b_pow_sum += b * b                                                                                              
            if b_pow_sum == 0 or a_pow_sum == 0:                                                                            
                centroid = 0                                                
            else:                                                                   
                centroid = (np.sqrt(b_pow_sum) / np.sqrt(a_pow_sum)) * (self.fs/np.pi)
        return centroid

    def contrast(self):
        """Computes the Spectral Contrast from magnitude spectrum. It returns the contrast and the valleys""" 
        min_real = 1e-30
        part_to_scale = 0.85                                                            
        bin_width = (self.fs/2) / self.M                                                      
        last_bins = round(20/bin_width)                                                 
        start_at_bin = last_bins                                                                                             
        n_bins_in_bands = np.zeros(6)                                                                                        
        n_bins = round(11000 / bin_width)                                                                                    
        upperBound = round(part_to_scale*n_bins) * bin_width                                                                 
        staticBinsPerBand = round((1-part_to_scale)*n_bins / 6);                                                             
        ratio = upperBound / 20;                                                                                             
        ratioPerBand = pow(ratio, (1.0/6));                                                                                  
        currFreq = 20                                                                                                        
        n_bins_in_bands = np.int32(n_bins_in_bands)                                                                          
        self.sc = np.zeros(6)                                                                                                
        self.valleys = np.zeros(6) 
        spec_index = start_at_bin                                                                                          
        for i in range(6):                                                                                                   
            currFreq = currFreq * ratioPerBand                                                                               
            n_bins_in_bands[i] = int(round(currFreq / bin_width - last_bins + staticBinsPerBand))          
            last_bins = round(currFreq/bin_width)                                                                            
            spec_bin = start_at_bin 
        self.contrast_bands = []                                                                                      
        for band_index in range(n_bins_in_bands.size):                                                                       
            if spec_index < self.magnitude_spectrum.size:                                                                    
                band_mean = 0                                                                                                
                for i in range(int(n_bins_in_bands[band_index])):                                                            
                    if spec_index+i < self.magnitude_spectrum.size:                                                          
                        band_mean += self.magnitude_spectrum[spec_index+i]                                                    
                if n_bins_in_bands[band_index] != 0:                                                                         
                    band_mean /= n_bins_in_bands[band_index]                                                                 
                band_mean += min_real  
                self.contrast_bands.append(range(int(spec_index),int(min(spec_index + n_bins_in_bands[band_index], self.magnitude_spectrum.size))))
                self.magnitude_spectrum[int(spec_index):int(min(spec_index + n_bins_in_bands[band_index], self.magnitude_spectrum.size))] = sorted(self.magnitude_spectrum[spec_index:min(spec_index + n_bins_in_bands[band_index], self.magnitude_spectrum.size)])                       
                nn_bins = round(0.4* n_bins_in_bands[band_index])                                                     
                if nn_bins < 1:                                                                                       
                    nn_bins = 1                                                                                       
                sigma = 0 
                for i in range(int(nn_bins)):                                                                                
                    if spec_index + i < self.magnitude_spectrum.size:                                                         
                        sigma += self.magnitude_spectrum[spec_index+i]                                                        
                valley = sigma/nn_bins + min_real                                                                            
                sigma = 0                                                                                                    
                for i in range(n_bins_in_bands[band_index]):
                    if i > n_bins_in_bands[band_index]-nn_bins and spec_index+i-1 < self.magnitude_spectrum.size and i > 0:       
                        i -= 1                                                                                    
                    sigma += self.magnitude_spectrum[spec_index+i-1]                                                                        
                peak = sigma/nn_bins + min_real                                                                       
                self.sc[band_index] = (-1. * (pow(float(peak)/valley, 1./np.log(band_mean))))                                
                self.valleys[band_index] = np.log(valley)                                                             
                spec_index += n_bins_in_bands[band_index] 
        return self.sc, self.valleys

    def Loudness(self):
        """Computes Loudness of a frame""" 
        self.loudness = pow(sum(abs(self.frame)**2), 0.67)

    def spectral_peaks(self):
        """Computes magnitudes and frequencies of a frame of the input signal by peak interpolation"""
        thresh = np.where(self.magnitude_spectrum[1:-1] > 0, self.magnitude_spectrum[1:-1], 0)            
        next_minor = np.where(self.magnitude_spectrum[1:-1] > self.magnitude_spectrum[2:], self.magnitude_spectrum[1:-1], 0)
        prev_minor = np.where(self.magnitude_spectrum[1:-1] > self.magnitude_spectrum[:-2], self.magnitude_spectrum[1:-1], 0)
        peaks_locations = thresh * next_minor * prev_minor
        self.peaks_locations = peaks_locations.nonzero()[0] + 1
        val = self.magnitude_spectrum[self.peaks_locations]
        lval = self.magnitude_spectrum[self.peaks_locations -1]
        rval = self.magnitude_spectrum[self.peaks_locations + 1]
        iploc = self.peaks_locations + 0.5 * (lval- rval) / (lval - 2 * val + rval)
        self.magnitudes = val - 0.25 * (lval - rval) * (iploc - self.peaks_locations)
        self.frequencies = (self.fs/2) * iploc / self.N 
        bound = np.where(self.frequencies < 5000) #only get frequencies lower than 5000 Hz
        self.magnitudes = self.magnitudes[bound][:100] #we use only 100 magnitudes and frequencies
        self.frequencies = self.frequencies[bound][:100]
        self.frequencies, indexes = np.unique(self.frequencies, return_index = True)
        self.magnitudes = self.magnitudes[indexes]

    def fundamental_frequency(self):
        """Compute the fundamental frequency of the frame frequencies and magnitudes"""
        f0c = np.array([])
        for i in range(len(self.frequencies)):
            for j in range(0, (4)*int(self.frequencies[i]), int(self.frequencies[i])): #list possible candidates based on all frequencies
                if j < 5000 and j != 0 and int(j) == int(self.frequencies[i]):
                    f0c = np.append(f0c, self.frequencies[i])
 
        p = 0.5                    
        q = 1.4 
        r = 0.5               
        rho = 0.33                            
        Amax = max(self.magnitudes) 
        maxnpeaks = 10                
        harmonic = np.matrix(f0c)                          
        ErrorPM = np.zeros(harmonic.size)
        MaxNPM = min(maxnpeaks, self.frequencies.size)

        for i in range(0, MaxNPM):                                                     
            difmatrixPM = harmonic.T * np.ones(self.frequencies.size)            
            difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1)) * self.frequencies)                                         
            FreqDistance = np.amin(difmatrixPM, axis=1)                                                   
            peakloc = np.argmin(difmatrixPM, axis=1)             
            Ponddif = np.array(FreqDistance) * (np.array(harmonic.T) ** (-p))                                               
            PeakMag = self.magnitudes[peakloc]                                  
            MagFactor = 10 ** ((PeakMag - Amax) * 0.5)
            ErrorPM = ErrorPM + (Ponddif + MagFactor * (q * Ponddif - r)).T
            harmonic += f0c

        ErrorMP = np.zeros(harmonic.size)   
        MaxNMP = min(maxnpeaks, self.frequencies.size)               
        for i in range(0, f0c.size):                                                               
            nharm = np.round(self.frequencies[:MaxNMP] / f0c[i])
            nharm = (nharm >= 1) * nharm + (nharm < 1)                
            FreqDistance = abs(self.frequencies[:MaxNMP] - nharm * f0c[i])
            Ponddif = FreqDistance * (self.frequencies[:MaxNMP] ** (-p))    
            PeakMag = self.magnitudes[:MaxNMP]                                  
            MagFactor = 10 ** ((PeakMag - Amax) * 0.5)
            ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor * (q * Ponddif - r)))

        Error = (ErrorPM[0] / MaxNPM) + (rho * ErrorMP / MaxNMP)                                              
        f0index = np.argmin(Error)        
        self.f0 = f0c[f0index] 

    def harmonic_peaks(self):
      N = 20 if len(self.frequencies) >= 20 else len(self.frequencies)              
      self.harmonic_frequencies = np.zeros(N)
      self.harmonic_magnitudes = np.zeros(N)
      candidates = [[-1,0] for i in range(len(self.frequencies))]
      for i in range(self.frequencies.size):    
          ratio = self.frequencies[i] / self.f0                      
          harmonicNumber = round(ratio)                                      
                                        
          if len(self.frequencies) < harmonicNumber:
                harmonicNumber = len(self.frequencies)
                                     
          distance = abs(ratio - harmonicNumber)                             
          if distance <= 0.2 and ratio <= (N + 0.2) and harmonicNumber>0:         
            if (candidates[int(harmonicNumber-1)][0] == -1 or distance < candidates[int(harmonicNumber-1)][1]): # first occured candidate or a better candidate for harmonic                                                    
                candidates[int(harmonicNumber-1)][0] = i                     
                candidates[int(harmonicNumber-1)][1] = distance              
                                                                             
          elif distance == candidates[int(harmonicNumber-1)][1]:        # select the one with max amplitude
            if self.magnitudes[i] > self.magnitudes[int(candidates[int(harmonicNumber-1)][0])]:                                             
                candidates[int(harmonicNumber-1)][0] = i       
                candidates[int(harmonicNumber-1)][1] = distance
      for i in range(N):
            j = candidates[i][0]
            if j < 0:
                self.harmonic_frequencies[i] = j+1 * self.f0
                self.harmonic_magnitudes[i] = 0
            else:
                self.harmonic_frequencies[i] = self.frequencies[i]
                self.harmonic_magnitudes[i] = self.magnitudes[i]
      self.harmonic_frequencies, indexes = np.unique(self.harmonic_frequencies, return_index = True)
      self.harmonic_magnitudes = self.harmonic_magnitudes[indexes]
      n = 0          
      for i in range(self.harmonic_magnitudes.size):
          try:
              if self.harmonic_magnitudes[i] == 0: #ideally it's not the f0
                  self.harmonic_magnitudes = np.delete(self.harmonic_magnitudes, i)
                  self.harmonic_frequencies = np.delete(self.harmonic_frequencies, i)
          except IndexError:
              n += 1
              if self.harmonic_magnitudes[i-n] == 0:                                
                  self.harmonic_magnitudes = np.delete(self.harmonic_magnitudes, i-n)  
                  self.harmonic_frequencies = np.delete(self.harmonic_frequencies, i-n)
              break
      return self.harmonic_frequencies, self.harmonic_magnitudes

    def inharmonicity(self):
        """Computes the Inharmonicity in a signal as a relation of weigthed magnitudes and frequencies of harmonics"""
        num = 0
        den = pow(self.harmonic_magnitudes[1], 2)
        f0 = self.harmonic_frequencies[1]
        for i in range(1,len(self.harmonic_frequencies)):
            ratio = round(self.harmonic_frequencies[i]/f0)
            num += abs(self.harmonic_frequencies[i] - ratio * f0) * pow(self.harmonic_magnitudes[i],2)
            den += pow(self.harmonic_magnitudes[i], 2)
        if den == 0:
            return 1
        else:
            return num/(den*f0)

    def detect_by_polar(self):
        targetPhase = np.real(2*self.phase - self.phase)
        targetPhase = np.mod(targetPhase + np.pi, -2 * np.pi) + np.pi;
        distance = np.abs(np.array([cmath.polar(self.magnitude_spectrum[i] + (self.phase[i]-targetPhase[i]*1j)) for i in range(len(targetPhase))]))
        return distance.sum();             
        
    def dissonance(self):
        """Computes the Dissonance based on sensory and consonant relationships between pairs of frequencies"""
        a_weight_factor = a_weight(self.frequencies) 
        loudness = self.magnitudes
        loudness *= pow(a_weight_factor,2) 
        if sum(loudness) == 0:                              
            return 0
        total_diss = 0                 
        for p1 in range(len(self.frequencies)):       
            if (self.frequencies[p1] > 50):           
                bark_freq = bark(self.frequencies[p1])
                freq_init = bark_to_hz(bark_freq) - 1.18      
                freq_exit = bark_to_hz(bark_freq) + 1.18      
                p2 = 0                                        
                peak_diss = 0                           
                while (p2 < len(self.frequencies) and self.frequencies[p2] < freq_init and self.frequencies[p2] < 50):
                        p2 += 1                                       
                while (p2 < len(self.frequencies) and self.frequencies[p2] < freq_exit and self.frequencies[p2] < 10000):
                        d = 1.0 - consonance(self.frequencies[p1], self.frequencies[p2])                                    
                        if (d > 0):                                   
                            peak_diss += d*loudness[p2] + loudness[p1] / sum(loudness)                                                      
                        p2 += 1                                       
                partial_loud = loudness[p1] / sum(loudness)           
                if (peak_diss > partial_loud):             
                    peak_diss = partial_loud
                total_diss += peak_diss
        total_diss = total_diss / 2
        return total_diss

    def hfc(self): 
        """Computes the High Frequency Content"""
        bin2hz = (self.fs/2.0) / (self.magnitude_spectrum.size - 1)                   
        return (sum([i*bin2hz * pow(self.magnitude_spectrum[i],2) for i in range(len(self.magnitude_spectrum))])) * self.H

    def NoiseAdder(self, array):                 
        random.seed(0)                                              
        noise = np.zeros(len(array))                            
        for i in range(len(noise)):                                   
            noise[i] = array[i] + 1e-10 * (random.random()*2.0 - 1.0)              
        return noise

    def LPC(self):
        fft = self.fft(self.windowed_x)
        self.Phase(fft)
        self.Spectrum()
        invert = self.ISTFT(self.magnitude_spectrum)
        invert = np.array(invert).T                                    
        self.correlation = util.normalize(invert, norm = np.inf)
        subslice = [slice(None)] * np.array(self.correlation).ndim
        subslice[0] = slice(self.windowed_x.shape[0])
        self.correlation = np.array(self.correlation)[subslice]      
        if not np.iscomplexobj(self.correlation):               
            self.correlation = self.correlation.real #compute autocorrelation of the frame
        self.correlation.flags.writeable = False
        E = np.copy(self.correlation[0])
        corr = np.copy(self.correlation)
        p = 14
        reflection = np.zeros(p)
        lpc = np.zeros(p+1)
        lpc[0] = 1
        temp = np.zeros(p)
        for i in range(1, p+1):
            k = float(self.correlation[i])
            for j in range(i):
                k += self.correlation[i-j] * lpc[j]
            k /= E
            reflection[i-1] = k
            lpc[i] = -k
            for j in range(1,i):
                temp[j] = lpc[j] - k * lpc[i-j]
            for j in range(1,i):
                lpc[j] = temp[j]
            E *= (1-pow(k,2))
        return lpc[1:]

class sonify(MIR):                                
    def __init__(self, audio, fs):
        super(sonify, self).__init__(audio, fs)
    def filter_by_valleys(self):
        mag_y = np.copy(self.magnitude_spectrum)
        for i in range(len(self.valleys)):
            bandpass = (hann(len(self.contrast_bands[i])-1) * np.exp(self.valleys[i])) - np.exp(self.valleys[i]) #exponentiate to get a positive magnitude value
            mag_y[self.contrast_bands[i][0]:self.contrast_bands[i][-1]] += bandpass #a small change can make a big difference, they said
        return mag_y 
    def reconstruct_signal_from_mfccs(self):
        """Listen to the MFCCs in a signal by reconstructing the fourier transform of the mfcc sequence"""
        bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(self.filter_coef.T, self.filter_coef), axis=0))
        recon_stft = bin_scaling * np.dot(self.filter_coef.T,db2amp(self.table.T[:13].T.dot(self.mfcc_seq)))
        output = self.ISTFT(recon_stft)
        return output
    def climb_hills(self):
        """Do hill climbing to find good HFCs"""
        moving_points = np.arange(len(self.filtered))
        stable_points = []
        while len(moving_points) > 0:
            for (i, x) in reversed(list(enumerate(moving_points))):
                if x > 0 and x < len(self.filtered) - 1:
                    if self.filtered[x] >= self.filtered[x - 1] and self.filtered[x] >= self.filtered[x + 1]:
                        stable_points.append(x)
                        moving_points = np.delete(moving_points, i)
                    elif self.filtered[x] < self.filtered[x - 1]:
                        moving_points[i] -= 1
                    else:
                        moving_points[i] += 1
                elif x == 0:
                    if self.filtered[x] >= self.filtered[x + 1]:
                        stable_points.append(x)   
                        moving_points = np.delete(moving_points, i)
                    else:                                          
                        moving_points[i] += 1
                else:                        
                    if self.filtered[x] >= self.filtered[x - 1]:
                        stable_points.append(x)   
                        moving_points = np.delete(moving_points, i)
                    else:                                          
                        moving_points[i] -= 1  
        self.Filtered = [0] * len(self.filtered)         
        for x in set(stable_points):      
            self.Filtered[x] = self.filtered[x]
    def create_clicks(self, locations):
        """Create a clicking signal at specified sample locations"""
        angular_freq = 2 * np.pi * 1000 / float(self.fs)            
        click = np.logspace(0, -10, num=int(np.round(self.fs * 0.015)), base=2.0)                                                          
        click *= np.sin(angular_freq * np.arange(len(click)))                             
        click_signal = np.zeros(len(self.signal), dtype=np.float32)
        for start in locations:
            end = start + click.shape[0]     
            if end >= len(self.signal):       
                click_signal[start:] += click[:len(self.signal) - start]                                                                  
            else:                                                   
                click_signal[start:end] += click                    
        return click_signal
    def output_array(self, array):
        self.M = 1024
        self.H = 512
        output = np.zeros(len(self.signal)) 
        i = 0
        for frame in self.FrameGenerator():
            output[i*(self.M-self.H):(i*(self.M-self.H) + self.M)] += array[i]  
            i += 1
        return output
    def sonify_music(self):
        """Function to sonify MFCC, Spectral Contrast, HFC and Tempo of the input signal"""
        self.mfcc_outputs = []
        self.contrast_outputs = []
        self.hfcs = []
        self.f0s = []
        self.ats = []
        for frame in self.FrameGenerator():    
            self.window() 
            self.Spectrum()
            self.Phase(self.fft(self.windowed_x))
            self.MFCC_seq()
            self.mfcc_outputs.append(self.reconstruct_signal_from_mfccs())
            self.contrast()
            fft_filter = self.filter_by_valleys()
            self.contrast_outputs.append(self.ISTFT(fft_filter))
            self.hfcs.append(self.hfc())
            self.spectral_peaks()
            self.fundamental_frequency()
            self.f0s.append(self.f0)
        self.mfcc_outputs = self.output_array(self.mfcc_outputs)
        self.contrast_outputs = self.output_array(self.contrast_outputs)
        self.f0 = Counter(self.f0s).most_common()[0][0]
        self.mel_bands_global()
        self.bpm()
        self.tempo_onsets = clicks(frames = self.ticks * self.fs, length = len(self.signal), sr = self.fs, hop_length = self.H)
        starts = self.ticks * self.fs
        for i in starts:
            self.frame = self.signal[int(i):int(i)+1024]
            self.Envelope()
            self.AttackTime()
            self.ats.append(self.attack_time)
        starts = starts[np.where(self.ats < np.mean(self.ats))]
        self.ats = np.array(self.ats)[np.where(self.ats < np.mean(self.ats))]
        attacks = np.int64((np.array(self.ats) * self.fs) + starts)
        self.attacks = self.create_clicks(attacks) 
        self.hfcs /= max(self.hfcs)
        fir = firwin(11, 1.0 / 8, window = "hamming")  
        self.filtered = np.convolve(self.hfcs, fir, mode="same")
        self.climb_hills()
        self.hfc_locs = np.array([i for i, x in enumerate(self.Filtered) if x > 0]) * self.H
        self.hfc_locs = self.create_clicks(self.hfc_locs)

def danceability(audio, target, fs):        
    frame_size = int(0.01 * fs)
    audio = audio[:fs * 3]
    nframes = audio.size / frame_size
    S = []
    for i in range(int(nframes)):
        fbegin = i * frame_size
        fend = min((i+1) * frame_size, audio.size)
        S.append(np.std(audio[fbegin:fend]))
    for i in range(int(nframes)):
        S[i] -= np.mean(S) 
    for i in range(len(S)):      
        S[i] += S[i-1]           
    tau = []                     
    for i in range(310, 3300):   
        i *= 1.1              
        tau.append(int(i/10)) 
    tau = np.unique(tau) 
    F = np.zeros(len(tau))                                                            
    nfvalues = 0            
    for i in range(len(tau)):        
        jump = max(tau[i]/50, 1)                                                                                       
        if nframes >= tau[i]:                                                                              
            k = jump                  
            while k < int(nframes-tau[i]):
                fbegin = int(k)                        
                fend = int(k + tau[i])           
                w = attention_sgd(np.mat(S[fbegin:fend]),target[fbegin:fend])    
                reg = S[fbegin:fend] * w
                F[i] += np.sum((target[fbegin:fend]-reg)**2)
                k += jump        
            if nframes == tau[i]:
                F[i] = 0         
            else:              
                F[i] = np.sqrt(F[i] / ((nframes - tau[i])/jump)) 
            nfvalues += 1
        else:      
            break    

    dfa = np.zeros(len(tau))
    for i in range(nfvalues):
        if F[i+1] != 0:
            dfa[i] = np.log10(F[i+1] / F[i]) / np.log10( (tau[i+1]+3.0) / (tau[i]+3.0))
        else:
            break
    motion = dfa[np.nan_to_num(dfa) > 0]
    motion = motion[motion < 1.5]    
    return 1/(motion.sum() / len(motion))

def hpcp(song, nbins):
    def add_contribution(freq, mag, bounded_hpcp):
        for i in range(len(harmonic_peaks)):
            f = freq * pow(2., -harmonic_peaks[i][0] / 12.0)
            hw = harmonic_peaks[i][1]
            pcpSize = bounded_hpcp.size
            resolution = pcpSize / nbins
            pcpBinF = np.log2(f / ref_freq) * pcpSize
            leftBin = int(np.ceil(pcpBinF - resolution * M / 2.0))
            rightBin = int((np.floor(pcpBinF + resolution * M / 2.0)))
            assert(rightBin-leftBin >= 0)
            for j in range(leftBin, rightBin+1):
                distance = abs(pcpBinF - j)/resolution
                normalizedDistance = distance/M
                w = np.cos(np.pi*normalizedDistance)
                w *= w
                iwrapped = j % pcpSize
                if (iwrapped < 0):
                    iwrapped += pcpSize
                bounded_hpcp[iwrapped] += w * pow(mag,2) * hw * hw
        return bounded_hpcp
    precision = 0.00001
    M = 1 #one semitone as a window size M of value 1
    ref_freq = 440 #Hz
    nh = 12
    min_freq = 40
    max_freq = 5000
    split_freq = 500
    hpcp_LO = np.zeros(nbins)
    hpcp_HIGH = np.zeros(nbins)
    harmonic_peaks = np.vstack((np.zeros(nbins), np.zeros(nbins))).T
    size = nbins
    hpcps = np.zeros(size)
    for i in range(nh):
        semitone = 12.0 * np.log2(i+1.0)
        ow = max(1.0 , ( semitone /12.0)*0.5)
        while (semitone >= 12.0-precision):
            semitone -= 12.0
        for j in range(1, len(harmonic_peaks)-1):
            if (harmonic_peaks[j][0] > semitone-precision and harmonic_peaks[j][0] < semitone+precision):
                break
        if (harmonic_peaks[i][0] == harmonic_peaks[-1][0] and harmonic_peaks[i][1] == harmonic_peaks[-1][1] and i != 0 and i != (nh -1)):
            harmonic_peaks[i][0] = semitone
            harmonic_peaks[i][1] = 1/ow
        else:
            harmonic_peaks[i][1] += (1.0 / ow)
    for i in range(song.frequencies.size):
        if song.frequencies[i] >= min_freq and song.frequencies[i] <= max_freq:
            if not np.any(song.frequencies < 500):
                pass
            else:
                if song.frequencies[i] < split_freq:
                    hpcp_LO = add_contribution(song.frequencies[i], song.magnitudes[i], hpcp_LO)
            if song.frequencies[i] > split_freq:
                hpcp_HIGH = add_contribution(song.frequencies[i], song.magnitudes[i], hpcp_HIGH)
    if np.any(song.frequencies < 500):
        hpcp_LO /= hpcp_LO.max()
    else:
        hpcp_LO = np.zeros(nbins)
    hpcp_HIGH /= hpcp_HIGH.max()
    for i in range(len(hpcps)):
        hpcps[i] = hpcp_LO[i] + hpcp_HIGH[i]
    return hpcps

def addContributionHarmonics(pitchclass, contribution, M_chords):                                                        
  weight = contribution                                                                                                  
  for index_harm in range(1,12):                                                                                          
    index = pitchclass + 12 *np.log2(index_harm)                                                                          
    before = np.floor(index)                                                                                             
    after = np.ceil(index)                                                                                               
    ibefore = np.mod(before,12.0)                                                                                        
    iafter = np.mod(after,12.0)                                                                                          
    # weight goes proportionally to ibefore & iafter                                                                     
    if ibefore < iafter:                                                                                                 
        distance_before = index-before                                                                                   
        M_chords[int(ibefore)] += pow(np.cos(0.5*np.pi*distance_before),2)*weight                                        
        distance_after = after-index                                                                                     
        M_chords[int(iafter)] += pow(np.cos(0.5*np.pi*distance_after ),2)*weight                                         
    else:                                                                                                                
        M_chords[int(ibefore)] += weight                                                                                                                          
    weight *= .74
    return M_chords                                                                                                      
                                                                                                   
def addMajorTriad(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       
  # Major 3rd                                                                                                            
  third = root + 4                                                                                                       
  if third > 11:                                                                                                         
    third -= 12                                                                                                          
  M_chords = addContributionHarmonics(third, contribution, M_chords)
                                                                                                                           
  # Perfect 5th                                                                                                          
  fifth = root + 7                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)
  return M_chords                                                                

def addAugTriad(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       
  # Major 3rd                                                                                                            
  third = root + 4                                                                                                       
  if third > 11:                                                                                                         
    third -= 12                                                                                                          
  M_chords = addContributionHarmonics(third, contribution, M_chords)
                                                                                                                           
  # augmented                                                                                                          
  fifth = root + 8                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)  
  return M_chords      
 
def addMinorSeventhTriad(root, contribution, M_chords):                                                          
  # Root                                                                                                                 
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                 
                                                                                                                         
  # Minor 3rd                                                                                                            
  third = root+3                                                                                                         
  if third > 11:                                                                                                         
    third -= 12                                                                    
  M_chords = addContributionHarmonics(third, contribution, M_chords)                          
  # Perfect 5th                                                                    
  fifth = root+7                                                                   
  if fifth > 11:                                                                   
    fifth -= 12                                                                    
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)  
  # 7th                                                                    
  seventh = root+11                                                                   
  if seventh > 11:                                                                   
    seventh -= 12                                                                    
  M_chords = addContributionHarmonics(seventh, contribution, M_chords)   
  return M_chords      
 
def addSus4Triad(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       
  # 4th                                                                                                           
  third = root + 5                                                                                                       
  if third > 11:                                                                                                         
    third -= 12                                                                                                          
  M_chords = addContributionHarmonics(third, contribution, M_chords)
                                                                                                                           
  # 3rd note (third from scale is suspended)                                                                                                        
  fifth = root + 7                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)  
  return M_chords      

def addSus2Triad(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       
  # 4th                                                                                                           
  second = root + 2                                                                                                       
  if second > 11:                                                                                                         
    second -= 12                                                                                                          
  M_chords = addContributionHarmonics(second, contribution, M_chords)
                                                                                                                           
  # 3rd note (third from scale is suspended)                                                                                                        
  fifth = root + 7                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)  
  return M_chords      

def addJazzChords(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       

  # 2nd                                                                                                       
  second = root + 2                                                                                                       
  if second > 11:                                                                                                         
    second -= 12                                                                                                          
  M_chords = addContributionHarmonics(second, contribution, M_chords) 

  # 4th                                                                                                           
  third = root + 5                                                                                                       
  if third > 11:                                                                                                         
    third -= 12                                                                                                          
  M_chords = addContributionHarmonics(third, contribution, M_chords)
                                                                                                                           
  #5th note (third from scale is suspended)                                                                                                        
  fifth = root + 7                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)  
                                                                                                        
  ninth = root + 10                                                                                                       
  if ninth > 11:                                                                                                         
    ninth -= 12                                                                                                          
  M_chords = addContributionHarmonics(ninth, contribution, M_chords)   
  return M_chords      
  
def addDimTriad(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       
  # diminished 3rd                                                                                                            
  third = root + 3                                                                                                       
  if third > 11:                                                                                                         
    third -= 12                                                                                                          
  M_chords = addContributionHarmonics(third, contribution, M_chords)
                                                                                                                           
  # diminished 5th                                                                                                          
  fifth = root + 6                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords)
  return M_chords      
  
def addThirteenthChord(root, contribution, M_chords):                                                          
  # Root            
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                                                                       

  extend = root + 2                                                                                                      
  if extend > 11:                                                                                                         
    extend -= 12                                                                                                          
  M_chords = addContributionHarmonics(extend, contribution, M_chords)  
                                                                                                    
  extend = root + 4                                                                                                      
  if extend > 11:                                                                                                         
    extend -= 12                                                                                                          
  M_chords = addContributionHarmonics(extend, contribution, M_chords)  

  extend = root + 5                                                                                                       
  if extend > 11:                                                                                                         
    extend -= 12                                                                                                          
  M_chords = addContributionHarmonics(extend, contribution, M_chords) 
                                                                                                                                                                                                                                   
  fifth = root + 7                                                                                                       
  if fifth > 11:                                                                                                         
    fifth -= 12                                                                                                          
  M_chords = addContributionHarmonics(fifth, contribution, M_chords) 
 
  extend = root + 9                                                                                                       
  if extend > 11:                                                                                                         
    extend -= 12                                                                                                          
  M_chords = addContributionHarmonics(extend, contribution, M_chords) 
  
  extend = root + 11                                                                                                       
  if extend > 11:                                                                                                         
    extend -= 12                                                                                                          
  M_chords = addContributionHarmonics(extend, contribution, M_chords)         

  return M_chords      
                                                                                                                    
def addMinorTriad(root, contribution, M_chords):                                                                         
  # Root                                                                                                                 
  M_chords = addContributionHarmonics(root, contribution, M_chords)                                                                 
                                                                                                                         
  # Minor 3rd                                                                                                            
  third = root+3                                                                                                         
  if third > 11:                                                                                                         
    third -= 12                                                                    
  M_chords = addContributionHarmonics(third, contribution, M_chords)                          
  # Perfect 5th                                                                    
  fifth = root+7                                                                   
  if fifth > 11:                                                                   
    fifth -= 12                                                                    
  M_chords = addContributionHarmonics(fifth, contribution, M_chords) 
  return M_chords       

def correlation(v1, mean1, std1, v2, mean2, std2, shift):                                                                
  v1 = (v1 - mean1) / (std1 * len(v1))  
  v2 = (v2 - mean2) / std2
  return np.correlate(v1,np.roll(v2,shift))[0]                                                                                          

def Key(pcp):                                                                                                            
    slope = 0.6                                                                                                          
    #M,m,aug,sus(Jazz),sus4,sus2
    tuller_hordiales =  np.array([[0.06540852, 0.09295253, 0.00188105,1., 0., 0.07970044, 0.11365712, 0.10877776,
       0.36621089, 0.02502536, 0.469842  , 0.20950151],
    [0.04754683,0.09765843, 0.02959059,1., 0.09489373, 0.03009842, 0.37268564, 0.,
       0.33602228, 0.05863082, 0.58744394, 0.23305194], 
    [0.71396226, 0.03195886, 0.14465802, 0.00201328,1. , 0.17706491, 0.06556756, 0.01620899, 0.2448181 , 0.39539417, 0.07605821, 0.],
    [0.33547322, 0.10670692, 0.5693731,1., 0.10097367, 0.0977697 , 0., 0.68889616,
       0.46660727, 0.14499055, 0.40802432, 0.30845697],
    [0.04768732,0.12992613, 0.,1, 0.0121699 , 0.06682322, 0.10715992, 0.02968415,
       0.46392702, 0.00938089, 0.40636099, 0.191441],
    [0.08617162,0.05722011, 0.,1., 0.29250521, 0.0995357 , 0.14099503, 0.05379608,
       0.30089182, 0.17968443, 0.71029348, 0.13515619],
    [0.03500999,0.18056602, 0.0720791,1., 0.09513679, 0., 0.61223628, 0.01521482,0.36155314, 0.03164523, 0.69254728, 0.25139205],
    [ 0.04476661,0.14460977, 0.09164541,1., 0.14328982, 0.04717347, 0.34793803, 0.,0.38140704, 0.22036064, 0.04030427, 0.3093726],
    [0.1950194 , 0.62064613, 0.,1.,0.46397092, 0.10135025, 0.28827229, 0.23941266, 0.57719603,0.23877516, 0.22102566,0.56707421]])                    
    M_chords = np.zeros(pcp.size)                                                                                              
    m_chords = np.zeros(pcp.size) 
    aug_chords = np.zeros(pcp.size)                                                                                              
    jazz_chords = np.zeros(pcp.size) 
    sus4_chords = np.zeros(pcp.size) 
    sus2_chords = np.zeros(pcp.size)  
    m7_chords = np.zeros(pcp.size) 
    dim_chords = np.zeros(pcp.size)    
    thirteenth_chords = np.zeros(pcp.size)                                                                                         
    key_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"] #it is good to remember keys sometimes 
    _M = tuller_hordiales[0]                                                                                                       
    _m = tuller_hordiales[1] 
    _aug = tuller_hordiales[2]
    _jazz = tuller_hordiales[3] 
    _sus4 = tuller_hordiales[4]
    _sus2 = tuller_hordiales[5]
    _m7 = tuller_hordiales[6] 
    _dim = tuller_hordiales[7] 
    _13 = tuller_hordiales[8]    
    #Do-Re-Mi-Fa-Sol-La-Si                                                                                                     
    M_chords = addMajorTriad(0, _M[0], M_chords) #first argument accounts for key index, not scale degree, so first of all the tonic   
    M_chords = addContributionHarmonics(1, _M[1], M_chords) 
    M_chords = addMinorTriad(2, _M[2], M_chords) 
    M_chords = addContributionHarmonics(3, _M[3], M_chords) 
    M_chords = addMinorTriad(4, _M[4], M_chords) #third   
    M_chords = addMajorTriad(5, _M[5], M_chords)   
    M_chords = addContributionHarmonics(6, _M[6], M_chords)         
    M_chords = addMajorTriad(7, _M[7], M_chords) #dominant (the famous 5th)
    M_chords = addContributionHarmonics(8, _M[8], M_chords)  
    M_chords = addContributionHarmonics(9, _M[9], M_chords) 
    M_chords = addContributionHarmonics(10, _M[10], M_chords) 
    M_chords = addContributionHarmonics(11, _M[11], M_chords)                             
    #Do,Re, Re#, Fa,Fa#, La, La#, Si
    dim_chords = addDimTriad(0, _dim[0], dim_chords) 
    dim_chords = addContributionHarmonics(1, _dim[1], dim_chords) 
    dim_chords = addDimTriad(2, _dim[2], dim_chords) 
    dim_chords = addMinorTriad(3, _dim[3], dim_chords)  
    dim_chords = addContributionHarmonics(4, _dim[4], dim_chords)    
    dim_chords = addMajorTriad(5, _dim[5], dim_chords)             
    dim_chords = addAugTriad(6, _dim[6], dim_chords)   
    dim_chords = addContributionHarmonics(7, _dim[7], dim_chords) 
    dim_chords = addContributionHarmonics(8, _dim[8], dim_chords)
    dim_chords = addContributionHarmonics(9, _dim[9], dim_chords) 
    dim_chords = addContributionHarmonics(10, _dim[10], dim_chords) 
    dim_chords = addContributionHarmonics(11, _dim[11], dim_chords)        
    #Do-Re-Re#-Fa-Sol-La-Si                                 
    m_chords = addMinorTriad(0, _m[0], m_chords) #tonic of minor chord    
    m_chords = addContributionHarmonics(1, _m[1], m_chords)   
    m_chords = addDimTriad(2, _m[2], m_chords)     
    m_chords = addContributionHarmonics(3, _m[3], m_chords)        
    m_chords = addAugTriad(4, _m[4], m_chords) #subdominant of minor chord   
    m_chords = addMinorTriad(5, _m[5], m_chords) 
    m_chords = addContributionHarmonics(6, _m[6], m_chords) 
    m_chords = addMajorTriad(7, _m[7], m_chords) #subdominant of minor chord            
    m_chords = addContributionHarmonics(8, _m[8], m_chords) 
    m_chords = addContributionHarmonics(9, _m[9], m_chords) 
    m_chords = addContributionHarmonics(10, _m[10], m_chords)   
    m_chords = addContributionHarmonics(11, _m[11], m_chords)     
    #Do-Re#-Mi-Sol-Sol# 
    aug_chords = addAugTriad(0, _aug[0], aug_chords) 
    aug_chords = addContributionHarmonics(1, _aug[1], aug_chords) 
    aug_chords = addContributionHarmonics(2, _aug[2], aug_chords) 
    aug_chords = addAugTriad(3, _aug[3], aug_chords) 
    aug_chords = addAugTriad(4, _aug[4], aug_chords) 
    aug_chords = addContributionHarmonics(5, _aug[5], aug_chords)  
    aug_chords = addContributionHarmonics(6, _aug[6], aug_chords)   
    aug_chords = addContributionHarmonics(7, _aug[7], aug_chords)                
    aug_chords = addMajorTriad(8, _aug[8], aug_chords) 
    aug_chords = addContributionHarmonics(9, _aug[9], aug_chords) 
    aug_chords = addContributionHarmonics(10, _aug[10], aug_chords)  
    aug_chords = addContributionHarmonics(11, _aug[11], aug_chords)    
    #Do-Re-Fa-Sol-Si
    jazz_chords = addJazzChords(0, _jazz[0], jazz_chords) 
    jazz_chords = addContributionHarmonics(1, _jazz[1], jazz_chords)
    jazz_chords = addMinorTriad(2, _jazz[2], jazz_chords)
    jazz_chords = addContributionHarmonics(3, _jazz[3], jazz_chords)
    jazz_chords = addContributionHarmonics(4, _jazz[4], jazz_chords)   
    jazz_chords = addSus2Triad(5, _jazz[5], jazz_chords)       
    jazz_chords = addContributionHarmonics(6, _jazz[6], jazz_chords)         
    jazz_chords = addMajorTriad(7, _jazz[7], jazz_chords) 
    jazz_chords = addContributionHarmonics(8, _jazz[8], jazz_chords)
    jazz_chords = addContributionHarmonics(9, _jazz[9], jazz_chords)
    jazz_chords = addContributionHarmonics(10, _jazz[10], jazz_chords)
    jazz_chords = addDimTriad(11, _jazz[11], jazz_chords)    
    #Do-Re-Fa-Sol-La# 
    sus4_chords = addSus4Triad(0, _sus4[0], sus4_chords) 
    sus4_chords = addContributionHarmonics(1, _sus4[1], sus4_chords) 
    sus4_chords = addMinorTriad(2, _sus4[2], sus4_chords) 
    sus4_chords = addContributionHarmonics(3, _sus4[3], sus4_chords) 
    sus4_chords = addContributionHarmonics(4, _sus4[4], sus4_chords) 
    sus4_chords = addSus2Triad(5, _sus4[5], sus4_chords)     
    sus4_chords = addContributionHarmonics(6, _sus4[6], sus4_chords)               
    sus4_chords = addMajorTriad(7, _sus4[7], sus4_chords) 
    sus4_chords = addContributionHarmonics(8, _sus4[8], sus4_chords)  
    sus4_chords = addContributionHarmonics(9, _sus4[9], sus4_chords) 
    sus4_chords = addMajorTriad(10, _sus4[10], sus4_chords)  
    sus4_chords = addContributionHarmonics(11, _sus4[11], sus4_chords)                   
    sus2_chords = addSus2Triad(0, _sus2[0], sus2_chords) 
    sus2_chords = addContributionHarmonics(1, _sus2[1], sus2_chords) 
    sus2_chords = addMinorTriad(2, _sus2[2], sus2_chords)     
    sus2_chords = addContributionHarmonics(3, _sus2[3], sus2_chords)    
    sus2_chords = addContributionHarmonics(4, _sus2[4], sus2_chords)   
    sus2_chords = addSus2Triad(5, _sus2[5], sus2_chords)    
    sus2_chords = addContributionHarmonics(6, _sus2[6], sus2_chords)       
    sus2_chords = addMajorTriad(7, _sus2[7], sus2_chords) 
    sus2_chords = addContributionHarmonics(8, _sus2[8], sus2_chords) 
    sus2_chords = addContributionHarmonics(9, _sus2[9], sus2_chords) 
    sus2_chords = addMajorTriad(10, _sus2[10], sus2_chords)  
    sus2_chords = addContributionHarmonics(11, _sus2[11], sus2_chords)         
    m7_chords = addMinorSeventhTriad(0, _m7[0], m7_chords) 
    m7_chords = addContributionHarmonics(1, _m7[1], m7_chords) 
    m7_chords = addDimTriad(2, _m7[2], m7_chords)
    m7_chords = addContributionHarmonics(3, _m7[3], m7_chords) 
    m7_chords = addAugTriad(4, _m7[4], m7_chords)    
    m7_chords = addMinorTriad(5, _m7[5], m7_chords)    
    m7_chords = addContributionHarmonics(6, _m7[6], m7_chords)             
    m7_chords = addMajorTriad(7, _m7[7], m7_chords)
    m7_chords = addContributionHarmonics(8, _m7[8], m7_chords)  
    m7_chords = addContributionHarmonics(9, _m7[9], m7_chords) 
    m7_chords = addContributionHarmonics(10, _m7[10], m7_chords)  
    m7_chords = addDimTriad(11, _m7[11], m7_chords)   
    #Do-Mi-Sol-Si-Re-Fa-La         
    thirteenth_chords = addThirteenthChord(0, _13[0], thirteenth_chords)
    thirteenth_chords = addContributionHarmonics(1, _13[1], thirteenth_chords)  
    thirteenth_chords = addMinorTriad(2, _13[2], thirteenth_chords) 
    thirteenth_chords = addContributionHarmonics(3, _13[3], thirteenth_chords) 
    thirteenth_chords = addMinorTriad(4, _13[4], thirteenth_chords) 
    thirteenth_chords = addMajorTriad(5, _13[5], thirteenth_chords)     
    thirteenth_chords = addContributionHarmonics(6, _13[6], thirteenth_chords)               
    thirteenth_chords = addMajorTriad(7, _13[7], thirteenth_chords)  
    thirteenth_chords = addContributionHarmonics(8, _13[8], thirteenth_chords)  
    thirteenth_chords = addMinorTriad(9, _13[9], thirteenth_chords)  
    thirteenth_chords = addContributionHarmonics(10, _13[10], thirteenth_chords) 
    thirteenth_chords = addDimTriad(11, _13[11], thirteenth_chords)  
    for n in range(12): 
        dominant = n+7;
        if dominant > 11:
            dominant -= 12
        M_chords[n]= _M[n] + (1.0/3.0)*_M[dominant]
        m_chords[n]= _m[n] + (1.0/3.0)*_m[dominant]  
        m7_chords[n]= _m7[n] + (1.0/3.0)*_m7[dominant]
        dim_chords[n]= _dim[n] + (1.0/3.0)*_dim[dominant]
        sus4_chords[n]= _sus4[n] + (1.0/3.0)*_sus4[dominant]  
        sus2_chords[n]= _sus2[n] + (1.0/3.0)*_sus2[dominant]  
        jazz_chords[n]= _jazz[n] + (1.0/3.0)*_jazz[dominant] 
        thirteenth_chords[n]= _13[n] + (1.0/3.0)*_13[dominant]       
    _M = M_chords                                                 
    _m = m_chords 
    _aug = aug_chords 
    _jazz = jazz_chords 
    _sus4 = sus4_chords  
    _sus2 = sus2_chords               
    _m7 = m7_chords  
    _dim = dim_chords   
    _13 = thirteenth_chords                                                                               
    prof_dom = np.zeros(pcp.size)                                                  
    prof_doM = np.zeros(pcp.size) 
    prof_aug = np.zeros(pcp.size)
    prof_jazz = np.zeros(pcp.size) 
    prof_sus4 = np.zeros(pcp.size)  
    prof_sus2 = np.zeros(pcp.size)  
    prof_m7 = np.zeros(pcp.size)
    prof_dim = np.zeros(pcp.size)
    prof_13 = np.zeros(pcp.size)                                                 
    for i in range(12):                                                            
        prof_doM[int(i*(pcp.size/12))] = tuller_hordiales[0][i]                              
        prof_dom[int(i*(pcp.size/12))] = tuller_hordiales[1][i] 
        prof_aug[int(i*(pcp.size/12))] = tuller_hordiales[2][i] 
        prof_jazz[int(i*(pcp.size/12))] = tuller_hordiales[3][i] 
        prof_sus4[int(i*(pcp.size/12))] = tuller_hordiales[4][i] 
        prof_sus2[int(i*(pcp.size/12))] = tuller_hordiales[5][i]  
        prof_m7[int(i*(pcp.size/12))] = tuller_hordiales[6][i]  
        prof_dim[int(i*(pcp.size/12))] = tuller_hordiales[7][i] 
        prof_13[int(i*(pcp.size/12))] = tuller_hordiales[8][i]                                      
        if i == 11:                                                                
            incr_M = (_M[11] - _M[0]) / (pcp.size/12)                    
            incr_m = (_m[11] - _m[0]) / (pcp.size/12)  
            incr_aug = (_aug[11] - _aug[0]) / (pcp.size/12)  
            incr_jazz = (_jazz[11] - _jazz[0]) / (pcp.size/12)  
            incr_sus4 = (_sus4[11] - _sus4[0]) / (pcp.size/12)  
            incr_sus2 = (_sus2[11] - _sus2[0]) / (pcp.size/12) 
            incr_m7 = (_m7[11] - _m7[0]) / (pcp.size/12)  
            incr_dim = (_dim[11] - _dim[0]) / (pcp.size/12) 
            incr_13 = (_13[11] - _13[0]) / (pcp.size/12)                     
        else:                                                                      
            incr_M = (_M[i] - _M[i+1]) / (pcp.size/12)                   
            incr_m = (_m[i] - _m[i+1]) / (pcp.size/12)
            incr_aug = (_aug[i] - _aug[i+1]) / (pcp.size/12)  
            incr_jazz = (_jazz[i] - _jazz[i+1]) / (pcp.size/12)
            incr_sus4 = (_sus4[i] - _sus4[i+1]) / (pcp.size/12)
            incr_sus2 = (_sus2[i] - _sus2[i+1]) / (pcp.size/12) 
            incr_m7 = (_m7[i] - _m7[i+1]) / (pcp.size/12) 
            incr_dim = (_dim[i] - _dim[i+1]) / (pcp.size/12) 
            incr_13 = (_13[i] - _13[i+1]) / (pcp.size/12)                               
        for j in range(int(pcp.size/12)):                                             
            prof_dom[int(i*(pcp.size/12)+j)] = _m[i] - j * incr_m 
            prof_doM[int(i*(pcp.size/12)+j)] = _M[i] - j * incr_M
            prof_aug[int(i*(pcp.size/12)+j)] = _aug[i] - j * incr_aug
            prof_jazz[int(i*(pcp.size/12)+j)] = _jazz[i] - j * incr_jazz
            prof_sus4[int(i*(pcp.size/12)+j)] = _sus4[i] - j * incr_sus4  
            prof_sus2[int(i*(pcp.size/12)+j)] = _sus2[i] - j * incr_sus2  
            prof_m7[int(i*(pcp.size/12)+j)] = _m7[i] - j * incr_m7
            prof_dim[int(i*(pcp.size/12)+j)] = _dim[i] - j * incr_dim 
            prof_13[int(i*(pcp.size/12)+j)] = _13[i] - j * incr_13                          
    mean_profM = np.mean(prof_doM)                                                 
    mean_profm = np.mean(prof_dom)  
    mean_aug = np.mean(prof_aug) 
    mean_jazz = np.mean(prof_jazz)  
    mean_sus4 = np.mean(prof_sus4)  
    mean_sus2 = np.mean(prof_sus2)    
    mean_m7 = np.mean(prof_m7) 
    mean_dim = np.mean(prof_dim)
    mean_13 = np.mean(prof_13)                                          
    std_profM = np.std(prof_doM)                                                   
    std_profm = np.std(prof_dom)  
    std_aug = np.std(prof_aug)   
    std_jazz = np.std(prof_jazz) 
    std_sus4 = np.std(prof_sus4) 
    std_sus2 = np.std(prof_sus2) 
    std_m7 = np.std(prof_m7) 
    std_dim = np.std(prof_dim)
    std_13 = np.std(prof_13)                                                        
    mean = np.mean(pcp)  
    std = np.std(pcp)                       
    maxM = -1                                 
    max2M = -1                                            
    keyiM = 0           
    maxm = -1                                                       
    max2min = -1             
    keyim = 0  
    maxaug = -1                                                       
    max2aug = -1             
    keyiaug = 0 
    maxjazz = -1                                                       
    max2jazz = -1            
    keyijazz = 0 
    maxsus4 = -1                                                       
    max2sus4 = -1             
    keyisus4 = 0
    maxsus2 =-1                                                       
    max2sus2 = -1             
    keyisus2 = 0 
    maxm7 = -1                                                       
    max2m7 = -1             
    keyim7 = 0     
    maxdim = -1 
    max2dim = -1 
    keyidim = 0   
    max13 = -1 
    max213 = -1 
    keyi13 = 0                                        
    for shift in range(pcp.size):
        corrM = correlation(pcp, mean, std, prof_doM, mean_profM, std_profM, shift)
        pcpfM = pcp[shift] / (attention_sga(np.mat(pcp),prof_doM)*pcp).max() 
        corrm = correlation(pcp, mean, std, prof_dom, mean_profm, std_profm, shift)
        pcpfm = pcp[shift] / (attention_sga(np.mat(pcp),prof_dom)*pcp).max() 
        corraug = correlation(pcp, mean, std, prof_aug, mean_aug, std_aug, shift)
        pcpfaug = pcp[shift] / (attention_sga(np.mat(pcp),prof_aug)*pcp).max() 
        corrjazz = correlation(pcp, mean, std, prof_jazz, mean_jazz, std_jazz, shift)
        pcpfjazz = pcp[shift] /(attention_sga(np.mat(pcp),prof_jazz)*pcp).max() 
        corrsus4 = correlation(pcp, mean, std, prof_sus4, mean_sus4, std_sus4, shift)
        pcpfsus4 = pcp[shift] /(attention_sga(np.mat(pcp),prof_sus4)*pcp).max() 
        corrsus2 = correlation(pcp, mean, std, prof_sus2, mean_sus2, std_sus2, shift)
        pcpfsus2 = pcp[shift] /(attention_sga(np.mat(pcp),prof_sus2)*pcp).max() 
        corrm7 = correlation(pcp, mean, std, prof_m7, mean_m7, std_m7, shift)
        pcpfm7 = pcp[shift] /(attention_sga(np.mat(pcp),prof_m7)*pcp).max() 
        corrdim = correlation(pcp, mean, std, prof_dim, mean_dim, std_dim, shift)
        pcpfdim = pcp[shift] /(attention_sga(np.mat(pcp),prof_dim)*pcp).max() 
        corr13 = correlation(pcp, mean, std, prof_13, mean_13, std_13, shift)
        pcpf13 = pcp[shift] /(attention_sga(np.mat(pcp),prof_13)*pcp).max() 
        if .74 > pcpfM: 
            corrM *= (pcpfM/.74)
        if .74 > pcpfm: 
            corrm *= (pcpfm/.74)
        if .74 > pcpfaug: 
            corraug *= (pcpfaug/.74)
        if .74 > pcpfjazz: 
            corrjazz *= (pcpfjazz/.74)
        if .74 > pcpfsus4: 
            corrsus4 *= (pcpfsus4/.74)
        if .74 > pcpfsus2: 
            corrsus2 *= (pcpfsus2/.74)  
        if .74 > pcpfm7: 
            corrm7 *= (pcpfm7/.74)
        if .74 > pcpfdim: 
            corrdim *= (pcpfdim/.74)
        if .74 > pcpf13: 
            corr13 *= (pcpf13/.74)
        P = p(corrM,pcp.size)
        if corrM > maxM and .01 > P: 
            max2M = maxM           
            maxM = corrM 
            keyiM = shift
        P = p(corrm,pcp.size)
        if corrm > maxm and .01 > P:
            max2min = maxm
            maxm = corrm
            keyim = shift
        P = p(corraug,pcp.size)
        if corraug > maxaug and .01 > P:
            max2aug = maxaug
            maxaug = corraug
            keyiaug = shift  
        P = p(corrjazz,pcp.size)
        if corrjazz > maxjazz and .01 > P:
            max2jazz = maxjazz
            maxjazz = corrjazz
            keyijazz = shift  
        P = p(corrsus4,pcp.size)
        if corrsus4 > maxsus4 and .01 > P:
            max2sus4 = maxsus4
            maxsus4 = corrsus4
            keyisus4 = shift 
        P = p(corrsus2,pcp.size)
        if corrsus2 > maxsus2 and .01 > P:
            max2sus2 = maxsus2
            maxsus2 = corrsus2
            keyisus2 = shift 
        P = p(corrm7,pcp.size)
        if corrm7 > maxm7 and .01 > P:
            max2m7 = maxm7
            maxm7 = corrm7
            keyim7 = shift  
        P = p(corrdim,pcp.size)
        if corrdim > maxdim and .01 > P:
            max2dim = maxdim
            maxdim = corrdim
            keyidim = shift 
        P = p(corr13,pcp.size)
        if corr13 > max13 and .01 > P:
            max213 = max13
            max13 = corr13
            keyi13 = shift                                                                                    
    correlated = np.argmax([maxM,maxm,maxaug,maxjazz,maxsus4,maxsus2,maxm7,maxdim,max13])
    if correlated == 0:
        keyi = int(keyiM * 12 / pcp.size + .5)
        scale = 'MAJOR'
        maximum = maxM
        max2 = max2M
    if correlated == 1:
        keyi = int(keyim * 12 / pcp.size + .5)
        scale = 'MINOR'
        maximum = maxm
        max2 = max2min
    if correlated == 2:
        keyi = int(keyiaug * 12 / pcp.size + .5)
        scale = 'AUGMENTED'
        maximum = maxaug
        max2 = max2aug  
    if correlated == 3:
        keyi = int(keyijazz * 12 / pcp.size + .5)
        scale = 'JAZZ(9sus4)'
        maximum = maxjazz
        max2 = max2jazz 
    if correlated == 4:
        keyi = int(keyisus4 * 12 / pcp.size + .5)
        scale = 'SUSPENDED FOURTH'
        maximum = maxsus4
        max2 = max2sus4 
    if correlated == 5:
        keyi = int(keyisus2 * 12 / pcp.size + .5)
        scale = 'SUSPENDED SECOND'
        maximum = maxsus2
        max2 = max2sus2
    if correlated == 6:
        keyi = int(keyim7 * 12 / pcp.size + .5)
        scale = 'MINOR SEVENTH'
        maximum = maxm7
        max2 = max2m7  
    if correlated == 7:
        keyi = int(keyidim * 12 / pcp.size + .5)
        scale = 'DIMINISHED'
        maximum = maxdim
        max2 = max2dim 
    if correlated == 8:
        keyi = int(keyi13 * 12 / pcp.size + .5)
        scale = 'THIRTEENTH'
        maximum = max13
        max2 = max213  
    if keyi >= 12: #keyi % 12
        keyi -= 12                                                                     
    key = key_names[keyi]
            
    firstto_2ndrelative_strength = (maximum - max2) / maximum
    print(key,firstto_2ndrelative_strength)
    return key, scale, firstto_2ndrelative_strength

def chord_sequence(song,hpcps):
    chords = ChordsDetection(np.array(hpcps), song)
    return list(np.array(chords)[:,0])
    
def ChordsDetection(hpcps, song):
    nframesm = int((2 * song.fs) / song.H) - 1
    chords = [] 
    for i in range(nframesm):   
        istart = max(0, i - nframesm/2)
        iend = min(i + nframesm/2, hpcps.shape[0])
        mean_hpcp = np.mean(hpcps[int(istart):int(iend)], axis=0).T
        mean_hpcp /= mean_hpcp.max()
        if mean_hpcp.size < 12:
            continue
        key, scale, firstto_2ndrelative_strength = Key(mean_hpcp)
        if float(firstto_2ndrelative_strength) <= 0:
            continue
        chords.append((key + scale, firstto_2ndrelative_strength))
    return chords   
