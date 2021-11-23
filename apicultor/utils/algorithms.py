#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from scipy.signal import *
import numpy as np
import scipy
from soundfile import read
from scipy.fftpack import fft, ifft
from collections import Counter
from ..machine_learning.cache import memoize
from ..gradients.ascent import *
from ..gradients.descent import *
from ..sonification.Sonification import *
import random
import cmath
#from apicultor.machine_learning.quality import *

#presion_diferencial = max(presion_latidos) / presion_latidos (la presión se relaja según el esfuerzo máximo)
#aumentar la presión incrementa el BPM, pero el BPM no es causa del incremento de presión

p = lambda r, n: (r * np.sqrt(n-2)) / np.sqrt(1-(r**2))

hz2cent = lambda hz: np.floor(120 * np.log2(hz) + -693.2631656229591)

cent2hz = lambda cent, f0: f0 * pow(pow(2, 10 / 1200.0), cent)

#TODO: Sintesis de transientes: DCT->FFT->IFFT-IDCT

#SNR = energía_señal / energía_antiseñal (ruido), distancia / anti_distancia
#graves y agudos según distancia: un agudo debe estar un 70% mas lejos para escucharse a su intensidad real
#la importancia evolutiva es que escucharla de tan lejos evitaba el desapego
#80m = audible con atención
#>100m = imposible de oir

def dmean(data):
    derived = np.zeros(len(data))
    for i in range(len(data)-1):
        derived[i] = data[i+1] - data[i]
    return np.mean(derived)

def dmean2(data):     
    derived = np.zeros(len(data))
    derived2 = derived.copy()
    for i in range(len(data)-1):
        derived[i] = data[i+1] - data[i]
        derived2[i] = derived[i+1] - derived[i]; 
    return np.mean(derived2)

energy = lambda mag: np.sum((10 ** (mag / 20)) ** 2)  

def central_moments(self):
    mean = np.mean(self.frame)
    mean /= self.frame.size
    x = self.frame -mean
    x2 = x*x
    sum2 = x2
    sum3 = x2*x
    sum4 = x2**2
    central_moments = [0,1,sum2/self.frame.size,sum3/self.frame.size,sum4/self.frame.size]
    scale = 1/self.frame.size-1
    norm = sum(self.frame)
    if norm == 0:
        central_moments = np.zeros(5)
    centroid = 0
    for i in range(self.frame.size):
        centroid += i*scale*self.frame[i]
    centroid /= norm
    central_moments[0] = 1
    central_moments[1] = 0
    m2,m3,m4 = 0,0,0
    for i in range(self.frame.size):
        v = i*scale-centroid
        v2 = v**2
        v2f = v2*self.frame[i]
        m2+=v2f
        m3+=v2f*v
        m4+=v2f*v2
    m2/=norm
    m3/=norm
    m4/=norm
    r = 1
    central_moments[2] = m2*pow(r,2)
    central_moments[3] = m3*pow(r,3)
    central_moments[4] = m4*pow(r,4)
    return central_moments

def dist_shape(central_moments):
    spread = central_moments[2] #variance of residuals
    if spread is 0:
        skewness = 0
    else:
        skewness = central_moments[3] / pow(spread,1.5)
    if spread is 0:
        kurtosis = -3
    else:
        kurtosis = (central_moments[4] / (spread*spread))-3
    return spread, skewness, kurtosis

#complexity = mag.size

def intensity(complexity,kurtosis,dissonance):
    if np.mean(complexity) <= 12.717778:
        if dmean(complexity) <= 1.912363:
            intensity = -1
        else:
            if np.mean(kurtosis)<= 7.098977:
                intensity = 0
            else:
                intensity = -1
    else:
        if dmean2(dissonance) <= 0.04818:
            intensity = 1
        else:
            intensity = 0
    return ['relaxed','moderate','aggresive'][intensity]

def roll_off(spectrum,fs):
    e_m = energy(spectrum)
    cutoff = .85 * e_m
    cume = 0
    rolloff = 0
    for i in range(len(spectrum)):
        cume += spectrum[i]*spectrum[i]
        if cume >= cutoff:
            rolloff=i
    rolloff *= (fs/2) / (spectrum.size-1)
    return rolloff


def audio_fingerprint(peaks):
    encoded_peaks = hash(peaks[:4])
    return (encoded_peaks[3] - (encoded_peaks[3] % 2)) * 100000000 + (encoded_peaks[2] - (encoded_peaks[2] % 2)) * 100000 + (encoded_peaks[1] (encoded_peaks[1] % 2)) * 100 + (encoded_peaks[0] - (encoded_peaks[0] % 2))

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='bandstop')#, fs, )
        y = lfilter(b,a,data)
        return y

def desaturate_signal(song):
    o = np.zeros(song.signal.size)
    start = 0
    for frame in song.FrameGenerator():
        indexes = np.unique(find_saturation(song))
        if indexes != []:
            filtered = butter_bandstop_filter(song.frame,indexes.min(),indexes.max(),48000,1)
            o[start:song.frame.size+start] = filtered
        else:    
            o[start:song.frame.size+start] = song.frame
        start += song.H
    return o

def find_saturation(song):
    prev_start = 0
    idx = 0
    start_proc = int((song.frame.size / 2) - (song.H / 2))
    end_proc = int((song.frame.size / 2) + (song.H / 2))
    energy_thresh = -1
    diff_thresh = .001
    min_dur = .005
    if (start_proc) < 2:
        start_proc = 2
    delta = abs(song.frame[start_proc - 1] - song.frame[start_proc - 2])
    past_mask = song.frame[start_proc - 1] > energy_thresh and delta < diff_thresh
    uflanks = [] 
    dflanks = []
    starts = []
    ends = []
    if past_mask and not prev_start:
        uflanks.append(start_proc - 1)
    for i in range(start_proc,end_proc):
        delta = abs(song.frame[i] - song.frame[i-1])
        current_mask = song.frame[i] > energy_thresh and delta < diff_thresh
        if current_mask and not past_mask:
            uflanks.append(i)
        elif not current_mask and past_mask:
            dflanks.append(i)     
        past_mask = current_mask
    if prev_start and len(dflanks) > 9:
        start = prev_start
        end = float(idx * song.H + dflanks[0] / 48000)
        duration = end -start
        if duration > min_dur:
            starts.append(start)
            ends.append(end)
        prev_start = 0
        dflanks.remove(dflanks[0])
    if len(uflanks) != len(dflanks) and len(uflanks) > 0:
        prev_start = idx * song.H + uflanks[-1] / 48000
        uflanks.pop(-1) 
    if len(uflanks) != len(dflanks) and idx == 0:
        dflanks.pop(-1)
    for i in range(len(uflanks)):
        start = float(idx * song.H + uflanks[0] / 48000) 
        end = float(idx * song.H + dflanks[0] / 48000) 
        duration = end -start
        if duration >= min_dur:
            starts.append(start)
            ends.append(end)
    idx += 1
    return starts,ends


def mono_stereo(input_signal): 
    input_signal = input_signal.astype(np.float32)   
    if len(input_signal.shape) != 1:               
        input_signal = input_signal.sum(axis = 1)/2
        return input_signal
    else:                  
        return input_signal

def hanning_NSGCQ(_window,M):
  for i in range(int(M/2)):
    _window[i] = 0.5 + 0.5 * np.cos(2.0*np.pi*i / M);
  
  for i in range(int(M/2+1),int(M)):
    _window[i] = 0.5 + 0.5 * np.cos(-2.0*np.pi*i / M);
  
  return _window

db2amp = lambda value: 0.5*pow(10, value/10) 

m_to_hz = lambda m: 700.0 * (np.exp(m/1127.01048) - 1.0)

hz_to_m = lambda hz: 1127.01048 * np.log(hz/700.0 + 1.0)

a_weight = lambda f: 1.25893 * pow(12200,2) * pow(f, 4) / ((pow(f, 2) + pow(20.6, 2)) * (pow(f,2) + pow(12200,2)) * np.sqrt(pow(f,2) + pow(107.7,2)) * np.sqrt(pow(f,2) + pow(737.9,2))) #a weighting of frequencies

bark_critical_bandwidth = lambda bark: 52548.0 / (pow(bark,2) - 52.56 * bark + 690.39) #compute a critical bandwidth with Bark frequencies

rms = lambda signal: np.sqrt(np.mean(np.power(signal,2))) 

def strong_peak(spectrum):
    minmag = spectrum.min()
    maxmag = spectrum.max()
    if minmag is maxmag:
        return 0
    thres = maxmag / 2
    bandwidth_left = spectrum.argmax()
    while bandwidth_left >0 and spectrum[bandwidth_left] > thres:
        bandwidth_left -= 1    
    if bandwidth_left != 0: bandwidth_left += 1
    elif spectrum[0] < thres:
        bandwidth_left += 1
    bandwidth_right = spectrum.argmax()
    bandwidth_right += 1
    while (bandwidth_right < spectrum.size and spectrum[bandwidth_right] >= thres):
        strong_peak = maxmag / np.log10(bandwidth_right / bandwidth_left)
        bandwidth_right += 1
    return strong_peak

def enhance_harmonics(_in):
    o = []
    for i in range(int(_in.size/4)):
        o.append(_in[i] +  _in[2*i] + _in[i*4])
    o = np.array(o) 
    _in[:o.size] = o          
    return _in

def trim_beats(localscore, beats, trim):
    """Final post-processing: throw out spurious leading/trailing beats"""

    smooth_boe = scipy.signal.convolve(localscore[beats],
                                       scipy.signal.hann(5),
                                       'same')

    if trim:
        threshold = 0.5 * ((smooth_boe**2).mean()**0.5)
    else:
        threshold = 0.0

    valid = np.argwhere(smooth_boe > threshold)

    return beats[valid.min():valid.max()]

def last_beat(cumscore):
    """Get the last beat from the cumulative score array"""

    maxes = np.max(cumscore)
    med_score = np.median(cumscore[np.argmax(cumscore)])

    # The last of these is the last beat (since score generally increases)
    return np.argwhere((cumscore * maxes * 2 > med_score)).max()

def true_peak_detector(signal):
    def correct_peaks(array):
        polef= 20000
        zerof=14.1e-3
        rpole =1-4*polef / signal.fs
        rzero = 1-4 * zerof / signal.fs
        b = np.array([0,-rzero])
        a = np.array([1,rpole])
    
        oversampling_factor = 4
        thres = -.0002
        filtered = lfilter(b,a,array)
        return filtered
    o = np.zeros(song.signal.size)
    start = 0
    for frame in song.FrameGenerator():
        o[start:song.frame.size+start] = correct_peaks(song.frame)
        start += song.H
    return o / o.max()


def NSGConstantQ(signal):                       
    fftres = signal.fs / 4096;
    nf = signal.fs / 2
    Q = pow(2,(1/48)) - pow(2,(-1/48)); 
    _gamma = 0         
    _minFrequency = 65.41
    b = int(np.floor(48 * np.log2(6000/65))); 
    _baseFreqs = np.zeros(b + 1);                       
                      
    cqtbw = np.zeros(b + 1);                         
    for j in range(int(b)-1):                         
        _baseFreqs[j] = (_minFrequency * pow(2,j / 48)) ;
        cqtbw[j] = Q * _baseFreqs[j] + _gamma;
    _binsNum = _baseFreqs.size  
    _baseFreqs = np.append(_baseFreqs,0)     
    _baseFreqs = np.append(_baseFreqs,22050) 

    for j in reversed(range(_binsNum)):                                                 
         _baseFreqs = np.append(_baseFreqs,signal.fs -_baseFreqs[j])

    bw = []                 
    bw.append(2*_minFrequency)
    bw.insert(0,cqtbw[-1])
    #bw.append(_baseFreqs[_binsNum+2] - _baseFreqs[_binsNum-1])
    for j in list(reversed(range(cqtbw.size-1))):
        bw.append(cqtbw[j])  
    
    _baseFreqs /= fftres
    bw = np.array(bw) / fftres

    posit = np.zeros(_baseFreqs.size);

    for j in range(_binsNum+1): 
      posit[j] = np.floor(_baseFreqs[j])
    for j in range(_binsNum+2,_baseFreqs.size): 
      posit[j] = np.ceil(_baseFreqs[j])

    _shifts = np.zeros(_baseFreqs.size);
    _shifts[0] = np.fmod( - posit[_baseFreqs.size-1] , 4096)

    for j in range(1,_baseFreqs.size): 
      _shifts[j] = posit[j] - posit[j-1]
      
    bw += 5

    _winsLen = np.zeros(_baseFreqs.size)
    _winsLen[:bw.size] = np.copy(bw)

    for j in range(_baseFreqs.size):
        if _winsLen[j] < 128:
            _winsLen[j] = 128

    _freqWins = [[] for i in range(_baseFreqs.size)]
    for j in range(_baseFreqs.size):
        inputWin = np.ones(_winsLen.size) 
        _freqWins[j] = hanning_NSGCQ(inputWin,int(_winsLen[j]))
    _freqWins = np.array(_freqWins)

    _windowSizeFactor = 1              
     
    _winsLen += -1                      
    _winsLen /= _windowSizeFactor    
    _winsLen += 1                    
                                 
    for j in range(int(_binsNum)+1):
      if ( _winsLen[j] > _winsLen[j+1] ):
          _freqWins[j] = _winsLen[j]
          _freqWins[j+1:] = _freqWins[j] + _winsLen[j]/2 - _winsLen[j+1]/2
          _freqWins[j] /= np.sqrt(_winsLen[j])   
    _binsNum = int(_baseFreqs.size / 2 - 1)

    normalizeWeights = np.ones(_binsNum+2)

    rasterizeIdx = _winsLen.size;

    for j in range(1,int(_binsNum)):
        rasterizeIdx -= 1;
        _winsLen[j] = _winsLen[_binsNum];
        _winsLen[rasterizeIdx] = _winsLen[_binsNum];

    for j in range(_winsLen.size):
        _winsLen[j] += (_winsLen[j] % 2)
    
    normalizeWeights[0] = _winsLen[0] + _binsNum+2
   
    normalizeWeights[0] *= normalizeWeights[-1] * 2 / 4096;

    for j in reversed(range(_binsNum)):
        normalizeWeights = np.append(normalizeWeights,normalizeWeights[j]);
    
    _freqWins *= normalizeWeights

    N = _shifts.size;

    fourier = np.fft.fft(signal.signal[:2048],4096)
  
    for i in list(reversed(range(int(4096/2-1)))):
      fourier[i] = np.conj(fourier[i])
    fill = _shifts[0] - 4096 ;

    posit.resize(N);
    posit[0] = _shifts[0];

    for j in range(1,N): 
      posit[j] = posit[j-1] + _shifts[j];
      
    fill += sum(_shifts[1:N])

    posit -= _shifts[0];

    #fourier = np.insert(fourier, np.zeros(fill));

    Lg = np.zeros(_freqWins.shape[0])
    for j in range(_freqWins.shape[0]):
      Lg[j] = _freqWins[j].size

      if ((posit[j] - Lg[j]/2) <= float(4096 + fill)/2):
        N = j+1;
 
    constantQ = [[] for i in range(N)]
    win_range = []
    for j in range(N):
      idx = []
      for i in range(int(np.ceil(Lg[j]/2.0))):
        if i < Lg[j]:
          idx.append(i);
      for i in range(int(np.ceil(Lg[j]/2))):
        idx.append(i);
      for i in range(int(-Lg[j]/2), int(np.ceil(Lg[j] / 2))):
        winComp = (posit[j] + i) % (4096 + fill);
        if (winComp >= fourier.size):
          winComp = (4096 + fill) - winComp;
        win_range.append( abs(winComp));
      product_idx = []
      for i in range(int(_winsLen[j] - (Lg[j] )/2),int(_winsLen[j] + int(Lg[j])/2 + .5)):      
                product_idx.append(np.fmod(i, _winsLen[j]));
      product = np.complex128(np.zeros(int(_winsLen[j])));
      for i in range(len(idx)):
              product[int(product_idx[i])] = fourier[int(win_range[i])] * _freqWins[j][idx[i]];
      displace = (posit[j] - ((posit[j] / _winsLen[j]) * _winsLen[j])) % len(product);
      product = np.roll(product,-1)
      constantQ[j] = np.fft.fft(product,int(_winsLen[j])) 
      constantQ[j] = list(reversed(constantQ[j]))
      constantQ[j] = constantQ[j] / len(constantQ[j])
      
    constantQ = np.array(constantQ)

    return constantQ,constantQ[0],constantQ[N-1:]


def dp(localscore, period, tightness):
    """Core dynamic program for beat tracking"""

    backlink = np.zeros_like(localscore, dtype=int)
    cumscore = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    # Make a score window, which begins biased toward start_bpm and skewed
    if tightness <= 0:
        raise ParameterError('tightness must be strictly positive')

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):

        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(- window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumscore[window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    return backlink, cumscore


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
        self.N = 4096
        self.M = 2048
        self.H = 1024
        self.n_bands = 40
        self.type = type
        self.nyquist = lambda hz: hz/(0.5*self.fs)
        self.to_db_magnitudes = lambda x: 20*np.log10(np.abs(x))
        self.duration = len(self.signal) / self.fs
        self.audio_signal_spectrum = []
        self.phase_signal = []
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

    def zcr(self):
          zero_crossing_rate = 0         
          if self.signal[0] < 1e-16:
              val = 0;
          else:
              val = self.signal[0]          
          was_positive = val > 0.0 
          for i in range(self.signal.size):
            val = self.signal[i];
            if abs(val) <= 1e-16:
                val = 0;
            is_positive = val > 0.0;    
            if (was_positive != is_positive):
              zero_crossing_rate += val
              was_positive = is_positive
          return zero_crossing_rate / self.signal.size

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
        if not type=='lowpass':              
            b,a = iirfilter(1, Wn, btype = type, ftype = 'butter')  
        else: 
            b,a = iirfilter(1, Wn, btype = type)       
        output = lfilter(b,a,array) #use a time-freq compromised filter
        return output 

    def pitch_salience_function(self,thres=1):
        _numberBins = int(np.floor(6000.0 / 10));
        _binsInSemitone = np.floor(100.0 / 10);
        _binsInOctave = 1200.0 / 10;
        _referenceTerm = 0.5 - _binsInOctave * np.log2(55);
        _magnitudeThresholdLinear = 1.0 / pow(10.0, thres/20.0);
        _nearestBinsWeights = np.zeros(int(_binsInSemitone)+1) 

        for b in range(int(_binsInSemitone)): 
            _nearestBinsWeights[b] = pow(np.cos((b/_binsInSemitone)* np.pi/2), 2); 
            
        _harmonicWeights = np.zeros(20)

        for h in range(20): 
            _harmonicWeights[h] = pow(.8,h); 

        if (self.harmonic_magnitudes.size != self.harmonic_frequencies.size): 
            raise IndexError("PitchSalienceFunction: frequency and magnitude input vectors must have the same size")
                
        if self.harmonic_frequencies == []:
            raise IndexError("Empty argument for frequencies")
                                    
        numberPeaks = len(self.harmonic_frequencies)
        for i in range(numberPeaks):
            if self.harmonic_frequencies[i] <= 0:
                raise ValueError("Frequencies must be positive")
            if self.harmonic_magnitudes[i] <= 0:
                raise ValueError("Magnitudes must be positive");         
            self.salience_function = np.zeros(_numberBins)
            minMagnitude = self.harmonic_magnitudes[np.argmax(self.harmonic_magnitudes)] * _magnitudeThresholdLinear;
            for i in range(numberPeaks):
                if self.harmonic_magnitudes[i] <= minMagnitude:
                    continue;
                magnitudeFactor = pow(self.harmonic_magnitudes[i], 1);
                for h in range(20):
                    h_bin = hz2cent(self.harmonic_frequencies[i] / (h+1));
                    if h_bin < 0:
                        break
                    for b in range(max(0,int(h_bin)-int(_binsInSemitone)),min(int(_numberBins)-1,int(h_bin)+int(_binsInSemitone))):
                        self.salience_function[b] += magnitudeFactor * _nearestBinsWeights[abs(b-int(h_bin))] * _harmonicWeights[h]

    def BandReject(self, array, cutoffHz, q):
        """Apply a 2nd order Infinite Impulse Response filter to the input signal  
        -param: cutoffHz: cutoff frequency in Hz                        
        -type: the type of filter to use [highpass, bandpass]"""  
        cutoffHz = np.array(cutoffHz)                                      
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
        output = lfilter(b,a,array) * 1
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
        output = lfilter(b, a, self.signal) * 1
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
        self.phase_signal.append(self.phase)

    def harmonic_distortion(self, fft,ref_frequency):
        """Computes harmonic distortion for a fourier transform""" 
        ref_harmonic = ref_frequency ** 2 
        sq_harmonics = np.sum([i** 2 for i in fft]) - ref_harmonic
        thd = 100 * sq_harmonics ** .5 / max(fft)
        return thd

    def Spectrum(self, fft=True):
        """Computes magnitude spectrum of windowed frame""" 
        if fft == True:                           
            self.spectrum = self.fft(self.windowed_x)                  
        else:                                                          
            self.magnitude_spectrum = constantq.cqt(self.windowed_x,hop_length=self.H, sr=self.fs,n_bins=113) #use constant Q(uality) for magnitude                                                                  
            self.audio_signal_spectrum.append(self.magnitude_spectrum) 
            return self                                                
        self.magnitude_spectrum = np.array([np.sqrt(pow(self.spectrum[i].real, 2) + pow(self.spectrum[i].imag, 2)) for i in range(self.H + 1)]).ravel() / self.H                                                
        self.audio_signal_spectrum.append(self.magnitude_spectrum)
    def spectrum_share(self):
        """Give back all stored computations of spectrums of different frames. This is a generator""" 
        for spectrum in self.audio_signal_spectrum:        
            self.magnitude_spectrum = spectrum 
            yield self.magnitude_spectrum

    def onsets_by_flux(self):
        """Use this function to get only frames containing peak parts of the signal""" 
        self.fluctuations = self.flux(self.mel_dbs)
        self.onsets_indexes = np.where(self.fluctuations > 80)[0]
        self.uncorrelated_indexes = np.where(self.fluctuations < 80)[0]
        self.audio_target_spectrum = np.array(self.audio_signal_spectrum)[self.onsets_indexes]
        self.target_mel_dbs = np.array(self.mel_dbs)[self.onsets_indexes]
        try:        
            self.phase_target = np.array(self.phase_signal)[self.onsets_indexes]
        except Exception as e:    
            pass
        self.frames_target_onset = np.array(list(self.FrameGenerator()))[self.onsets_indexes]
        self.audio_target_fluctuations = np.array(self.fluctuations)[self.onsets_indexes]
        self.uncorrelated_spectrum = np.array(self.audio_signal_spectrum)[self.uncorrelated_indexes]
        self.uncorrelated_frames_onset = np.array(list(self.FrameGenerator()))[self.uncorrelated_indexes]
        self.uncorrelated_fluctuations = np.array(self.fluctuations)[self.uncorrelated_indexes]
        self.uncorrelated_mel_dbs = np.array(self.mel_dbs)[self.onsets_indexes]
        try:        
            self.uncorrelated_phase = np.array(self.phase_signal)[self.uncorrelated_indexes]
        except Exception as e:    
            pass

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
            dct = np.zeros(len(array))
            for i in range(len(array)):
                dct[i] = 0
                for j in range(self.n_bands):
                    try:              	
                        dct[i] += array[j] * self.table[i][j]
                    except Exception as e: #array connections are smaller than the size of filter at array position
                        pass
        return dct

    def MFCC_seq(self):
        """Computes Mel Frequency Cepstral Coefficients. It returns the Mel Bands using a Mel filter and a sequence of MFCCs""" 
        self.mel_bands = self.MelFilter()
        self.mel_bands = self.mel_bands[self.mel_bands>0]
        dbs = 2 * (10 * np.log10(self.mel_bands))
        self.n_bands = len(dbs)
        self.mfcc_seq = self.DCT(dbs)

    def autocorrelation(self):
        self.N = (self.windowed_x[:,0].shape[0]+1) * 2
        corr = ifft(fft(self.windowed_x, n = self.N))  
        corr /= np.max(corr,axis=0,keepdims=True)
        self.correlation = corr
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
        self.envelope = np.mean(self.envelope, axis=0)
        pad_width = 1 + (2048//(2*self.H)) 
        try:     
            self.envelope = np.pad(self.envelope, ([0, 0], [int(pad_width), 0]),mode='constant') 
        except Exception as e:     
            self.envelope = np.pad([self.envelope], ([0, 0], [int(pad_width), 0]),mode='constant') 
        self.envelope = lfilter([1.,-1.], [1., -0.99], self.envelope, axis = -1) 
        self.envelope = self.envelope[:,:self.n_bands][0]  

    def bpm(self):   
        """Computes tempo of a signal in Beats Per Minute with its tempo onsets""" 
        self.onsets_strength()                                                             
        n = len(self.envelope) 
        win_length = np.asscalar(np.array([int(np.floor(8*48000/self.H))]))
        ac_window = hann(win_length) 
        self.envelope = np.pad(self.envelope, int(win_length // 2),mode='linear_ramp', end_values=[0, 0])
        frames = 1 + int((len(self.envelope) - win_length) / 1) 

        f = []                                                                            
        for i in range(win_length):     
            f.append(self.envelope[i:i+frames])
        f = np.array(f)[:,:n]              
        self.windowed_x = f * ac_window[:, np.newaxis]
        self.autocorrelation()

        tempogram = self.correlation

        bin_frequencies = np.zeros(int(tempogram.shape[0]), dtype=np.float)

        bin_frequencies[0] = np.inf
        bin_frequencies[1:] = 22 * self.fs / (self.H * np.arange(1.0, tempogram.shape[0]))

        prior = np.exp(-0.5 * ((np.log2(bin_frequencies) - np.log2(60)) / bin_frequencies[1:].std())**2)
        max_indexes = np.argmax(bin_frequencies < 208)
        min_indexes = np.argmax(bin_frequencies < 60)

        prior[:max_indexes] = 0
        prior[min_indexes:] = 0
        p = prior.nonzero()

        best_period = np.argmax(tempogram[p] * prior[p][:, np.newaxis] * -1, axis=0)
        self.tempo = bin_frequencies[p][best_period]

        period = round(60.0 * (self.fs/self.H) / self.tempo[0])

        window = np.exp(-0.5 * (np.arange(-period, period+1)*32.0/period)**2)
        localscore = convolve(self.envelope/self.envelope.std(ddof=1), window, 'same')
        backlink, cumscore = dp(localscore, period, 100)
        self.bpm = cumscore.max() * 2
        interv_value = int(np.floor(self.bpm / 60 * self.fs))
        interval = 0
        self.ticks = []
        for i in range(int(self.signal.size/interv_value)):
            self.ticks.append(interval + interv_value)
            interval += interv_value #compute tempo frames locations based on the beat location value
        self.ticks = np.array(self.ticks) / self.fs
        return self.bpm, self.ticks

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

    def find_peaks(self,function,peak_thres=0,interpolate=False,max_pts= 100):
        thresh = np.where(function[1:-1] > peak_thres, function[1:-1], 0)            
        next_minor = np.where(function[1:-1] > function[2:], function[1:-1], 0)
        prev_minor = np.where(function[1:-1] > function[:-2], function[1:-1], 0)
        peaks_locations = thresh * next_minor * prev_minor
        self.peaks_locations = peaks_locations.nonzero()[0] + 1
        if interpolate == True:
            val = function[self.peaks_locations]
            lval = function[self.peaks_locations -1] 
            rval = function[self.peaks_locations + 1]    
            iploc = self.peaks_locations + 0.5 * (lval- rval) / (lval - 2 * val + rval)
            return (val - 0.25 * (lval - rval) * (iploc - self.peaks_locations))[:max_pts]

        else:
            function[self.peaks_locations][:max_pts]

    def removePeak(self, peaksBins, peaksValues, i, j):  
        peakBins = np.delete(peaksBins[i], peaksBins[i][:j])
        peakValues = np.delete(peaksValues[i], peaksValues[i][:j])
        return peakBins, peakValues

    def trackPitchContour(self,_nonSalientPeaksBins,peak_thres=0):
        max_i = 0
        max_j = 0
        _timeContinuityInFrames = (100 / 1000.0) * 44100 / self.H
        maxSalience = 0
        bins = []
        saliences = []
        for i in range(_numberFrames):
            if (self.salience_values[i].size > 0):
                j = np.argmax(self.salience_values[i])
            if self.salience_values[i][j] > maxSalience:
                maxSalience = self.salience_values[j]
            max_i = i
            max_j = j
            if maxSalience == 0:
                return None
                
        index = max_i; #contour starts at maximum salience peak
        bins.append(self.salience_bins[index])  
        saliences.append(self.salience_values[index])   
        _salientPeaksBins, _salientPeaksValues = self.removePeak(_salientPeaksBins, _salientPeaksValues, index, max_j)   
        for i in range(_numberFrames):
            best_peak_j = self.findNextPeak(_salientPeaksBins, bins, i)
            if best_peak_j >= 0:
                bins.append(_salientPeaksBins[index][best_peak_j])
                saliences.append(_salientPeaksValues[index][best_peak_j])   
                _salientPeaksBins, _salientPeaksValues = self.removePeak(_salientPeaksBins, _salientPeaksValues, i, best_peak_j)
                gap = 0
            else:
                if gap+1 > _timeContinuityInFrames:
                    break
                best_peak_j = self.findNextPeak(_nonSalientPeaksBins, bins, i)
                if best_peak_j >= 0:
                    bins.append(_nonSalientPeaksBins[i][best_peak_j])
                    saliences.append(_nonSalientPeaksValues[i][best_peak_j])
                    removeNonSalientPeaks.append([i, best_peak_j])
                    gap += 1
                else:
                    break

        for g in range(gap):
            bins.pop(-1)
            saliences.pop(-1)
            if index == 0:
                if bins.size < _timeContinuityInFrames:
                    bins = np.array([])
                    return bins

        gap = 0
        for i in range(-1,size_t):
            best_peak_j = self.findNextPeak(_salientPeaksBins, bins, i)
            if best_peak_j >= 0:
                bins.append(_salientPeaksBins[index][best_peak_j])
                saliences.append(_salientPeaksValues[index][best_peak_j])   
                _salientPeaksBins, _salientPeaksValues = self.removePeak(_salientPeaksBins, _salientPeaksValues, i, best_peak_j) 
                i -= 1
                gap = 0
            else:
                if gap+1 > _timeContinuityInFrames:
                    break
                best_peak_j = self.findNextPeak(_nonSalientPeaksBins, bins, i)
                if best_peak_j >= 0:
                    bins.append(_nonSalientPeaksBins[i][best_peak_j])
                    saliences.append(_nonSalientPeaksValues[i][best_peak_j])
                    self.removeNonSalientPeaks.append([i, best_peak_j])
                    gap += 1
                else:
                    break
            if i >0:
                i -= 1
            else:
                break
                
        bins = np.delete(bins,bins[:gap])

        saliences = np.delete(saliences,saliences[:gap])
                
        index += gap

        for r in range(-1,self.removeNonSalientPeaks.size):
            i_p = self.removeNonSalientPeaks[r][0]
            if i_p < index or i_p > index + len(bins):
                continue
            j_p = self.removeNonSalientPeaks[r][1]
            bins, saliences = self.removePeak(_nonSalientPeaksBins, _nonSalientPeaksValues, i_p, j_p)

    def pitch_contours(self,peak_thres=0):
        """Computes magnitudes and frequencies of a frame of the input signal by peak interpolation"""
        _timeContinuityInFrames = (100 / 1000.0) * self.fs / self.H;
        _minDurationInFrames = (100 / 1000.0) * self.fs / self.H;
        _pitchContinuityInBins = 100 * 1000.0 * self.H / self.fs / 10;
        _frameDuration = self.H / self.fs
        _numberFrames = len(self.salience_bins)
        _nonSalientPeaksBins=np.zeros(_numberFrames)
        _nonSalientPeaksValues=np.zeros(_numberFrames)
        salientInFrame = []
        self.contour_bins = np.zeros(len(self.salience_bins))
        self.contour_saliences = np.zeros(len(self.salience_bins))
        for i in range(_numberFrames):
            numPeaks = self.salience_bins[i].size
        for i in range(_numberFrames):
            if (self.salience_values[i].size == 0):
                continue
            frameMinSalienceThreshold = .9 * max(self.salience_values[i]);    
            for j in range(self.salience_bins[i].size):
                if (self.salience_values[i][j] < frameMinSalienceThreshold):
                    _nonSalientPeaksBins[i] = self.salience_bins[i][j] 
                    _nonSalientPeaksValues[i] = self.salience_values[i][j]     
                else:
                    salientInFrame.append([i,j])

        for i in range(len(salientInFrame)):
            ii = salientInFrame[i][0]
            jj = salientInFrame[i][1]
            if (self.salience_values[ii][jj] < overallMeanSalienceThreshold):
                _nonSalientPeaksBins[ii] = self.salience_bins[ii][jj]
                _nonSalientPeaksValues[ii] = self.salience_values[ii][jj]
            else:
                _SalientPeaksBins[ii] = self.salience_bins[ii][jj]
                _SalientPeaksValues[ii] = self.salience_values[ii][jj]

        while True:
            bins, saliences = trackPitchContour(index, self.contour_bins, self.contour_saliences)
            jj = salientInFrame[i][1]
            if (self.contour_bins.size > 0):
                if (self.contour_bins.size >= _minDurationInFrames):
                    contoursStartTimes.append(i * _frameDuration)
                    self.contour_bins.append(bins);
                    self.contour_saliences.append(saliences)
                else:
                    break
                    
    def pitch_salience_function_peaks(self,peak_thres=0):
        """Computes magnitudes and frequencies of a frame of the input signal by peak interpolation"""
        self.find_peaks(self.salience_function,peak_thres=peak_thres)
        minBin = max(0.0, np.floor(10 * np.log2(55/55) + 0.5));
        maxBin = min(599,max(0.0, np.floor(10 * np.log2(1760/55) + 0.5)));
        peaks = self.salience_function[self.peaks_locations[self.peaks_locations>minBin]]
        #peaks = peaks[peaks<maxBin]
        bins = cent2hz(peaks,self.f0)
        self.salience_values = peaks
        self.salience_bins = bins

    def spectral_peaks(self):
        """Computes magnitudes and frequencies of a frame of the input signal by peak interpolation"""
        self.peak_thres = 0
        thresh = np.where(self.magnitude_spectrum[1:-1] > self.peak_thres, self.magnitude_spectrum[1:-1], 0)            
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
        bound = np.where(self.frequencies < 22000) #only get frequencies lower than 5000 Hz
        self.magnitudes = self.magnitudes[bound][:456] #we use only 100 magnitudes and frequencies
        self.frequencies = self.frequencies[bound][:456]
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

    #in lpc windowed signals can be filtered in such way that the predicted discrete signal = sum_of_p_coefficients(lpc*x+reflection)
    #so for each coefficient there is a reflection resulting in a multilinear problem
    def LPC(self):
        fft = self.fft(self.windowed_x)
        self.Phase(fft)
        self.Spectrum()
        invert = self.ISTFT(self.magnitude_spectrum)
        invert = np.array(invert).T                                    
        self.correlation = invert.T / invert.T.max()
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
        return lpc[1:],reflection

#MSA method

#spectral flux of mfccs
#pcp of constant q transforms

#similarity computation using cosine 

#community detection

#l = 1 
#diff_r increment of r steps

#w = profile of a beat
#w = w + r*l

#while abs(C*l) < len(beats):
#l = l+1
#Cl = {}
#iterate to find a subset of beats to include in Cl and break
#r = r+ diff_r
#w = w + r*l
#if we're gathering too many beats for the subsets the l would be large at some point

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

def width_interpolation(w_idx,size):
    w_interp = []
    for i in range(size):
        w_interp.append((w_idx[i]*( 1-((np.sin(2*np.pi*1/3)+1)/2.0) ) ) + ( w_idx[i-1]*((np.sin(2*np.pi*1/3)+1)/2.0)))
    return w_interp

def danceability(audio, w, cumsum, fs):        
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
    for i in range(11, 930):   
        i *= 1.5              
        tau.append(int(i/10)) 
    tau = np.unique(tau) 
    F = np.zeros(len(tau))                                                            
    nfvalues = 0   
    Target = width_interpolation(cumsum.cumsum(),len(S))         
    for i in range(len(tau)):        
        jump = max(tau[i]/50, 1)   
        if nframes >= tau[i]: 
            k = jump 
            while k < int(nframes-tau[i]):
                fbegin = int(k)
                fend = int(k + tau[i])   
                reg = S[fbegin:fend] * w
                F[i] += np.sum((Target[fbegin:fend]-reg)**2)
                k += jump     
            if nframes == tau[i]:
                F[i] = 0 
            else:
                F[i] = np.sqrt(F[i] / ((nframes - tau[i])/jump)) 
            nfvalues += 1
        else:
            break


    dfa = np.zeros(len(tau))
    for i in range(nfvalues-1):
        if F[i+1] != 0:
            dfa[i] = np.log10(F[i+1] / F[i]) / np.log10( (tau[i+1]+3.0) / (tau[i]+3.0))
        else:
            break
    motion = dfa[np.nan_to_num(dfa) > 0]
    motion = motion[motion < 1.5]    
    return 1/(motion.sum() / len(motion))

def music_structure_analysis(signal,separator,fun='median'):
    audio,fs = read(signal)
    audio = mono_stereo(audio)
    song = MIR(audio,fs)
    song.mel_bands_global() #cepstrum
    song.onsets_by_flux() #get the onsets based on fluctuation
    song.bpm() #compute bpm using cummulative score of a tempogram, maximum may not fit under a sample rate
    onsets_indexes = song.onsets_indexes * song.H
    ticks = song.ticks * song.fs #bars from BPM
    pcps = []
    for magnitude in song.audio_signal_spectrum:
        song.magnitude_spectrum = magnitude
        song.spectral_peaks()
        pcps.append(hpcp(song,28))
    pcps = np.array(pcps)
    ticks = np.array(ticks)    

    start = 0
    nodes = []
    Ticks = []
    for bar in range(len(ticks)-1):
        if any(nodes): #a max pitch profile had been found
            mfcc_nodes = onsets_indexes[(ticks[bar] < onsets_indexes)][(onsets_indexes[(ticks[bar] < onsets_indexes)] < ticks[bar+1])]
        else:
            conditioned_intensity = (onsets_indexes[(ticks[bar] < onsets_indexes)] < ticks[bar+1])
            try:
                mfcc_nodes
            except Exception as e:
                conditioned_intensity = (onsets_indexes[(ticks[bar] < onsets_indexes)] > ticks[bar+1])
                mfcc_nodes = onsets_indexes[(ticks[bar] < onsets_indexes)][conditioned_intensity] 
                first = True
            if not any(conditioned_intensity) or not np.any([mfcc_nodes]) and not first:
                conditioned_intensity = (onsets_indexes[(ticks[bar] < onsets_indexes)] > ticks[bar+1])
                mfcc_nodes = onsets_indexes[(ticks[bar] < onsets_indexes)][conditioned_intensity] 
        if type(mfcc_nodes) is list or type(mfcc_nodes) is np.ndarray:
            if not any(mfcc_nodes):
                continue
        if len(mfcc_nodes) == 1:   
            node_boundary = np.where(np.max(mfcc_nodes) == onsets_indexes)[0]
            if not any(node_boundary):
                continue
            mfcc_nodes = node_boundary[0]
            if not any(pcps[mfcc_nodes]):
                continue
        if len(np.ravel([mfcc_nodes])) > 1:
            node_boundary = np.where([i == onsets_indexes for i in mfcc_nodes])[1]
            if not any(node_boundary):
                continue
        if fun is 'median':
            if len(np.ravel([mfcc_nodes])) > 1:
                node_pcp = np.median(np.median(pcps[node_boundary],axis=1),axis=0)
            else:
                node_pcp = np.median(pcps[mfcc_nodes],axis=0)
        if fun is 'mean':
            if len(np.ravel([mfcc_nodes])) > 1:        
                node_pcp = np.mean(np.mean(pcps[node_boundary],axis=1),axis=0)
            else:
                node_pcp = np.mean(pcps[mfcc_nodes],axis=0)
        nodes.append(node_pcp)
        Ticks.append([ticks[bar],ticks[bar+1]])

    nodes = np.array(nodes) #the median of two bars is a node

    from sklearn.metrics import pairwise_distances 

    nans = np.isnan(nodes)
    valid = ~ nans    
    nodes = np.mat(nodes[np.unique(np.where(valid)[0])]).T
    ticks = ticks[np.unique(np.where(valid)[0])]
    cosine_similarities_nodes = pairwise_distances(nodes,metric='euclidean')

    l = 1    
    r = 1e-2
    W = cosine_similarities_nodes.copy()
    Cl = []
    regions = []
    W += (r * l)
    while sum([len(i) for i in Cl]) + 1 <= len(W):
        l += 1
        max_subset = np.argmax(W[:,sum([len(i) for i in Cl])][sum([len(i) for i in Cl]):]) + sum([len(i) for i in Cl])
        r += 1e-2
        W += (r * l)
        if np.any(Cl):
            delta = [max((np.max(Ticks[max_subset]),Cl[-1][0]))]
            r += np.max(Ticks[max_subset])/Cl[-1][0]
        else:
            delta = [np.max(Ticks[max_subset])]
        Cl.append(delta)
        
    Cl = np.unique(Cl)

    for i in range(len(Cl)):
        if i == 0:
            segment = audio[:int(Cl[i])] 
            if segment.size / song.fs > 2:
                write_file(separator+signal+str(i), song.fs,segment)            
        elif not i == len(Cl) -1:
            segment = audio[int(Cl[i-1]):int(Cl[i])] 
            if segment.size / song.fs > 2:
                write_file(separator+signal+str(i), song.fs,segment)     
        elif i == len(Cl) - 1:
            segment = audio[int(Cl[i]):] 
            if segment.size / song.fs > 2:
                write_file(separator+signal+str(i), song.fs,segment)     


    return Cl

def AMEKEQ200(song, db1,db2,db3,db4,db5,q1,q2,q3,q4,q5,hz1,hz2,hz3,hz4,hz5,band_mono, widening,thd_db):
    a = song.AllPass(hz1,q1) * (10 ** (db1/20))
    b = song.AllPass(hz2,q2) * (10 ** (db2/20))
    c = song.AllPass(hz3,q3) * (10 ** (db3/20))
    d = song.AllPass(hz4,q4) * (10 ** (db4/20))
    e = song.AllPass(hz5,q5) * (10 ** (db5/20))
    mid = 1 - ((widening) / 2)
    Mid = mid * (.5/mid)
    x_wide = []
    for i in range(len(song.signal)-2): 
        x_wide.append(song.signal[i] - Mid * song.signal[i+1] + song.signal[i+2])
    x_wide.append(0)
    x_wide.append(0)    
    x_wide = np.array(x_wide)    
    thd = (song.signal - np.mean(song.signal)) *  (10 ** (thd_db/20)) + song.signal
    mono = song.AllPass(band_mono,1)
    output = (a+b+c+d+e+thd+mono+x_wide)
    return output / output.max()



def hpcp(song, nbins):
    def add_contribution(freq, mag, bounded_hpcp):
        for i in range(len(harmonic_peaks)):
            f = freq * pow(2., -harmonic_peaks[i][0] / 12.0)
            hw = harmonic_peaks[i][1]
            pcpSize = bounded_hpcp.size
            resolution = pcpSize / 12
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
        hpcp_LO /= np.sum(hpcp_LO)
    else:
        hpcp_LO = np.zeros(nbins)
    hpcp_HIGH /= np.sum(hpcp_HIGH)
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
    tuller_hordiales =  np.array([[0.41195011, 0.3544457 , 0.33578951, 1.33864786, 0.49436168,0.65914687, 0.44601615, 0.5186047, 
    0.52313098, 0.36738215,1.36087088, 0.32623478],
    [0.07288526, 0.09463136, 0.0586082 , 0.52009415, 0.12960295,0.07560954, 0.19616892, 0.04831062, 0.20377285, 0.07926838,
       0.38413152, 0.13691624], 
    [0.37416135, 0.37555069, 0.23076177, 1.39934873, 0.68608055,0.25800434, 0.46741926, 0.72022854, 0.59505774, 0.44524911,
       0.16416206, 1.2545391 ],
    [0.10214688, 0.12714296, 0.04820773, 0.5099767 , 0.13532745,0.11133197, 0.11856532, 0.05864232, 0.26811043, 0.09039591,
       0.32439796, 0.10575436],
    [0.07103472, 0.27458088, 0.06249917, 0.37781489, 0.13657496,0.08395169, 0.25884847, 0.04931643, 0.17914072, 0.12303474,
       0.24107345, 0.14212987],
    [0.30132904, 0.59428861, 0.3251779 , 1.34322025, 0.51932745,0.20806407, 0.70662596, 0.43410324, 0.52365924, 0.79085678,
    0.22327312, 0.53492541]])                    
    M_chords = np.zeros(pcp.size)                                                                                              
    m_chords = np.zeros(pcp.size) 
    aug_chords = np.zeros(pcp.size)                                                                                              
    sus4_chords = np.zeros(pcp.size) 
    m7_chords = np.zeros(pcp.size) 
    dim_chords = np.zeros(pcp.size)                                                                                            
    key_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"] #it is good to remember keys sometimes 
    _M = tuller_hordiales[0]                                                                                                       
    _m = tuller_hordiales[1] 
    _aug = tuller_hordiales[2]
    _sus4 = tuller_hordiales[3]
    _m7 = tuller_hordiales[4] 
    _dim = tuller_hordiales[5]                                                                           
    prof_dom = _m                                                
    prof_doM = _M 
    prof_aug = _aug
    prof_sus4 = _sus4    
    prof_m7 = _m7
    prof_dim = _dim                                              
    #for i in range(12):                                                            
    #    prof_doM[int(i*(pcp.size/12))] = tuller_hordiales[0][i]                              
    #    prof_dom[int(i*(pcp.size/12))] = tuller_hordiales[1][i] 
    #    prof_aug[int(i*(pcp.size/12))] = tuller_hordiales[2][i] 
    #    prof_sus4[int(i*(pcp.size/12))] = tuller_hordiales[3][i] 
    #    prof_m7[int(i*(pcp.size/12))] = tuller_hordiales[4][i]  
    #    prof_dim[int(i*(pcp.size/12))] = tuller_hordiales[5][i]                                       
    #    if i == 11:                                                                
    #        incr_M = (_M[11] - _M[0]) / (pcp.size/12)                    
    #        incr_m = (_m[11] - _m[0]) / (pcp.size/12)  
    #        incr_aug = (_aug[11] - _aug[0]) / (pcp.size/12)  
    #        incr_sus4 = (_sus4[11] - _sus4[0]) / (pcp.size/12)  
    #        incr_m7 = (_m7[11] - _m7[0]) / (pcp.size/12)  
    #        incr_dim = (_dim[11] - _dim[0]) / (pcp.size/12)                    
    #    else:                                                                      
    #        incr_M = (_M[i] - _M[i+1]) / (pcp.size/12)                   
    #        incr_m = (_m[i] - _m[i+1]) / (pcp.size/12)
    #        incr_aug = (_aug[i] - _aug[i+1]) / (pcp.size/12)  
    #        incr_sus4 = (_sus4[i] - _sus4[i+1]) / (pcp.size/12)
    #        incr_m7 = (_m7[i] - _m7[i+1]) / (pcp.size/12) 
    #        incr_dim = (_dim[i] - _dim[i+1]) / (pcp.size/12)                             
    #    for j in range(int(pcp.size/12)):                                             
    #        prof_dom[int(i*(pcp.size/12)+j)] = _m[i] - j * incr_m 
    #        prof_doM[int(i*(pcp.size/12)+j)] = _M[i] - j * incr_M
    #        prof_aug[int(i*(pcp.size/12)+j)] = _aug[i] - j * incr_aug
    #        prof_sus4[int(i*(pcp.size/12)+j)] = _sus4[i] - j * incr_sus4  
    #        prof_m7[int(i*(pcp.size/12)+j)] = _m7[i] - j * incr_m7
    #        prof_dim[int(i*(pcp.size/12)+j)] = _dim[i] - j * incr_dim                         
    mean_profM = np.mean(prof_doM)                                                 
    mean_profm = np.mean(prof_dom)  
    mean_aug = np.mean(prof_aug) 
    mean_sus4 = np.mean(prof_sus4)  
    mean_m7 = np.mean(prof_m7) 
    mean_dim = np.mean(prof_dim)                                        
    std_profM = np.std(prof_doM)                                                   
    std_profm = np.std(prof_dom)  
    std_aug = np.std(prof_aug)   
    std_sus4 = np.std(prof_sus4) 
    std_m7 = np.std(prof_m7) 
    std_dim = np.std(prof_dim)                                                       
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
    maxsus4 = -1                                                       
    max2sus4 = -1             
    keyisus4 = 0
    maxm7 = -1                                                       
    max2m7 = -1             
    keyim7 = 0     
    maxdim = -1 
    max2dim = -1 
    keyidim = 0                                      
    for shift in range(pcp.size):
        corrM = correlation(pcp, mean, std, prof_doM, mean_profM, std_profM, shift)
        pcpfM = pcp[shift] / (attention_sga(np.mat(pcp),prof_doM)*pcp).max() 
        corrm = correlation(pcp, mean, std, prof_dom, mean_profm, std_profm, shift)
        pcpfm = pcp[shift] / (attention_sga(np.mat(pcp),prof_dom)*pcp).max() 
        corraug = correlation(pcp, mean, std, prof_aug, mean_aug, std_aug, shift)
        pcpfaug = pcp[shift] / (attention_sga(np.mat(pcp),prof_aug)*pcp).max() 
        corrsus4 = correlation(pcp, mean, std, prof_sus4, mean_sus4, std_sus4, shift)
        pcpfsus4 = pcp[shift] /(attention_sga(np.mat(pcp),prof_sus4)*pcp).max() 
        corrm7 = correlation(pcp, mean, std, prof_m7, mean_m7, std_m7, shift)
        pcpfm7 = pcp[shift] /(attention_sga(np.mat(pcp),prof_m7)*pcp).max() 
        corrdim = correlation(pcp, mean, std, prof_dim, mean_dim, std_dim, shift)
        pcpfdim = pcp[shift] /(attention_sga(np.mat(pcp),prof_dim)*pcp).max() 
        if .74 > pcpfM: 
            corrM *= (pcpfM/.74)
        if .74 > pcpfm: 
            corrm *= (pcpfm/.74)
        if .74 > pcpfaug: 
            corraug *= (pcpfaug/.74)
        if .74 > pcpfsus4: 
            corrsus4 *= (pcpfsus4/.74)
        if .74 > pcpfm7: 
            corrm7 *= (pcpfm7/.74)
        if .74 > pcpfdim: 
            corrdim *= (pcpfdim/.74)
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
        P = p(corrsus4,pcp.size)
        if corrsus4 > maxsus4 and .01 > P:
            max2sus4 = maxsus4
            maxsus4 = corrsus4
            keyisus4 = shift 
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
    correlated = np.argmax([maxM,maxm,maxaug,maxsus4,maxm7,maxdim])
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
        keyi = int(keyim7 * 12 / pcp.size + .5)
        scale = 'MINOR SEVENTH'
        maximum = maxm7
        max2 = max2m7  
    if correlated == 6:
        keyi = int(keyidim * 12 / pcp.size + .5)
        scale = 'DIMINISHED'
        maximum = maxdim
        max2 = max2dim 
    if keyi >= 12: #keyi % 12
        keyi -= 12                                                                     
    key = key_names[keyi]
            
    firstto_2ndrelative_strength = (maximum - max2) / maximum
    print(key,firstto_2ndrelative_strength)
    return key, scale, firstto_2ndrelative_strength

def chord_sequence(song,hpcps):
    chords = ChordsDetection(np.array(hpcps), song)
    return list(np.array(chords)[:,0])

def envelope_follower(x,fs,at=200,sus=500,rel=1000):
    at_coef = np.exp(np.log(0.01)/( at * fs * 0.001))
    sus_coef = np.exp(np.log(0.01)/(sus * fs * 0.001));  
    rel_coef = np.exp(np.log(0.01)/( rel * fs * 0.001));  
    envelope_frequencies = np.copy(x);
    tmp = np.abs(x);
    envelope[tmp>envelope_frequencies] = at_coef * (envelope[tmp>envelope_frequencies] - tmp) + tmp;
    envelope[tmp>envelope_frequencies] = sus_coef * (envelope[tmp>envelope_frequencies] - tmp) + tmp;
    envelope[tmp>envelope_frequencies] = rel_coef * (envelope[tmp>envelope_frequencies] - tmp) + tmp;
    return envelope
    
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
