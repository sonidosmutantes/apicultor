#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from librosa.core import stft
from sklearn.utils.extmath import safe_sparse_dot as ssd
from sklearn.cluster import AffinityPropagation as AP
from collections import Counter

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

stft_angle = lambda tf_map: np.arccos(tf_map)

def tf_map(real, imag):
    sqrt_i = np.sqrt(ssd(imag.T, imag))

    sqrt_r = np.sqrt(ssd(real.T, real))

    stft_k = ssd(real.T, imag)                     
    return stft_k / (sqrt_i * sqrt_r)  

def cos_dif(angle):
    cos = np.cos(np.diff(angle)) 

    dif = np.zeros(shape=angle.shape)
                     
    dif[:,1:cos.shape[0]] = cos

    return dif   

def find_sparse_source_points(real, imag):
    tf_plane = tf_map(real, imag)
    angle = stft_angle(tf_plane)
    dif = cos_dif(angle)
    X = angle.copy()
    is_sparse = np.abs(tf_plane) > dif                                                 
    for i in range(X.shape[0]):
        if np.any(is_sparse[i]) == True:        
            X[i][is_sparse[i] == True] = X[i][is_sparse[i] == True]
        else:
            X[i][is_sparse[i] == False] = 0
    X[np.isnan(X)] = 0
    return X

def cosine_distance(X):
    cos_dist = []
    for i in range(len(X)):
        cos_dist.append(1 - np.abs(tf_map(X.T[i- 1], X.T[i])))
    cos_dist = np.array(cos_dist)
    cos_dist[np.isnan(cos_dist)] = 0
    return cos_dist

def find_number_of_sources(cosine_distance):
    cos_dist = np.resize(cosine_distance, new_shape = (len(cosine_distance), len(cosine_distance))) 
    ap = AP(affinity = 'precomputed').fit(cos_dist)
    counter = Counter(ap.labels_).most_common()
    source = 0
    for i in range(len(counter)):
        if counter[i][1] == counter[0][1]:     
            source += 1
    return source


