#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re   
import json
import warnings
import numpy as np
from itertools import compress
import csv
import pandas

warnings.simplefilter("ignore", ResourceWarning)

def get_files(files_dir):
    """                                                           
    return a list of .json files to read
    :param files_dir: directory where sounds are located
    """
    try:                                                    
        for subdir, dirs, files in os.walk(files_dir+'/descriptores'):
            files_list = [f for f in files]
        return files_list                                     
    except:                                              
        if not os.path.exists(files_dir+'/descriptores'):
            print ("No .json files found in " + str(files_dir))


def get_dics(files_dir):
    """                                                           
    return a list containing all descriptions of tag directory reading the .json files
    :param files_dir: directory where sounds are located
    """
    dics = []
    try:                                        
        for subdir, dirs, files in os.walk(files_dir+'/descriptores'):
            for f in files:
                with open(subdir + '/' + f) as read:          
                    dics.append(json.load(read))
    except:                                              
        if not os.path.exists(files_dir+'/descriptores'):
            print ("No readable MIR data found")
    return dics

class desc_pair():                                     
    """                           
    obtain an array of descriptors                                                      
    :param files: list of files                    
    :param dics: list of .json files corresponding to files
    """          
    def __init__(self, files, dics):            
        self._files = files         
        self._dics = dics           
        self.descriptors = ['lowlevel.mfcc_bands.mean', 'metadata.duration.mean', 'lowlevel.hfc.mean', 'rhythm.bpm_ticks.mean', 'lowlevel.mfcc.mean', 'sfx.inharmonicity.mean', 'rhythm.bpm.mean', 'lowlevel.spectral_contrast.mean', 'sfx.logattacktime.mean', 'loudness.level.mean', 'lowlevel.spectral_valleys.mean', 'lowlevel.spectral_centroid.mean', 'lowlevel.dissonance.mean'] # modify according to features_and_keys

        selector = [np.any(np.isnan(np.float64(list(x.values())))) == False for x in self._dics]                        
        self._files_features = list(filter(lambda x: np.any(np.isnan(np.float64(list(x.values())))) == False, self._dics))
        indexes = list(compress(selector, self._dics))
        self._files = np.array(self._files)[indexes]

        self._features = []                
        for i in range(len(self._files_features)):
            self._features.append(list(self._files_features[i].values()))

        self._features = np.vstack(np.float64(self._features))
        self.keys = [i for i,x in enumerate(self.descriptors)] 

def read_file(f, data_dir):
    """ 
    Open the text file containing deep learning classification results for files. The decision variables is automatically opened from utils folder
    :param f: text filename          
    :returns:                                                                                                         
      - sounds: sounds that were analyzed
      - lc: labels for each sound
    """
    classes = pandas.read_csv(f).values
    sounds = np.array(classes)[:,1]
    lc = np.array(classes)[:,2]
    decisions = pandas.read_csv(data_dir+'/data.csv').values[:,(1,2,3,4)]
    return sounds, lc, decisions

def read_attention_file(data_dir):                 
    attention_variables = pandas.read_csv(data_dir+'/attention.csv').values.T[1:]
    return attention_variables, pandas.read_csv(data_dir+'/attention.csv').keys()

def read_good_labels(data_dir):                 
    good_labels = pandas.read_csv(data_dir+'/attention_labels.csv').values
    return good_labels
