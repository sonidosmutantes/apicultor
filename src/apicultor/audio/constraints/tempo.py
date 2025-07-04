#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
import librosa
from numpy.typing import NDArray


def same_time(x: NDArray[np.floating], y: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Synchronize two audio signals to have the same tempo.
    
    Args:
        x: First audio signal
        y: Second audio signal
        
    Returns:
        Tuple of synchronized audio signals with matching tempo and length
    """
    x_tempo = librosa.beat.beat_track(x)[0]                           
    y_tempo = librosa.beat.beat_track(y)[0]                           
    if x_tempo < y_tempo:                                             
        y = librosa.effects.time_stretch(y, x_tempo/y_tempo)           
        if y.size < x.size:                                            
            y = np.resize(y, x.size)                                   
        else:                                                         
            x = np.resize(x, y.size)                                  
    else:                                                             
        x = librosa.effects.time_stretch(x, y_tempo/x_tempo)          
        if x.size < y.size:                                            
            x = np.resize(x, y.size)                                   
        else:                                                          
            y = np.resize(y, x.size)                                   
    return x, y 
