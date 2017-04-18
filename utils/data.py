#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re   
import json
import numpy as np

def get_files(files_dir):
	"""
	return a list of .json files to read

	:param files_dir: directory where sounds are located
	"""
	try:
		files = [(files) for subdir, dirs, files in os.walk(files_dir+'/descriptores')]
		return files
	except:
		if not os.path.exists(files_dir+'/descriptores'):
			print ("No .json files found")

def get_dics(files_dir):
	"""
	return a list containing all descriptions of tag directory reading the .json files

	:param files_dir: directory where sounds are located
	"""
	try:
		for subdir, dirs, files in os.walk(files_dir+'/descriptores'):
			dics = [json.load(open(subdir+'/'+f)) for f in files]  
		return dics
	except:
		if not os.path.exists(files_dir+'/descriptores'):
			print ("No readable MIR data found")

class desc_pair():
	"""
	obtain description values after choosing descriptors

	:param desc1: first descriptor values
	:param desc2: second descriptor values
	:returns:
	  - euclidean_labels: labels of clusters
	""" 
	def __init__(self, files, dics):
		self._files = files
		self._dics = dics
		self.descriptors = ['lowlevel.dissonance.mean', 'lowlevel.mfcc_bands.mean', 'sfx.inharmonicity.mean', 'rhythm.bpm.mean', 'lowlevel.spectral_contrast.mean', 'lowlevel.spectral_centroid.mean', 'metadata.duration.mean', 'lowlevel.mfcc.mean', 'loudness.level.mean', 'rhythm.bpm_ticks.mean', 'lowlevel.spectral_valleys.mean', 'sfx.logattacktime.mean', 'lowlevel.hfc.mean'] # modify according to features_and_keys
		self.features = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(self._dics))  
		self.files_features = np.array_split(self.features, np.array(self._files).size) 
		self.keys = [i for i,x in enumerate(self.descriptors)]

def read_file(f):
    """ 
    Open the text file containing deep learning classification results for files. The decision variables is automatically opened from utils folder
    :param f: text filename          
    :returns:                                                                                                         
      - sounds: sounds that were analyzed
      - lc: labels for each sound
    """
    with open(f, 'r') as i_file:
        files = i_file.read().split('\n')
    sounds = np.array([i.split(" ")[0].split(".json")[0] for i in np.array(files)])
    labels = np.array([i.split(" ") for i in np.array(files)])
    lc = []
    for lb in labels:
        try: 
            lc.append(lb[1])
        except:
            continue
    decisions = np.loadtxt('utils/data.txt')
    return sounds, np.array(lc), decisions
