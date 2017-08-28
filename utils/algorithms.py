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
import random

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
