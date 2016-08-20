#! /usr/bin/env python
# -*- coding: utf-8 -*-


import time
from colorama import *
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                          
import json    
import re                                                         
from sklearn import preprocessing                                       
from sklearn.cluster import AffinityPropagation                                  
                                                                        
files_dir = 'descriptores/bajo'

descriptors = ['lowlevel.dissonance.mean', 'lowlevel.mfcc_bands.mean', 'sfx.inharmonicity.mean', 'rhythm.bpm.mean', 'lowlevel.spectral_contrast.mean', 'lowlevel.spectral_centroid.mean', 'lowlevel.mfcc.mean', 'loudness.level.mean', 'metadata.duration.mean', 'lowlevel.spectral_valleys.mean', 'sfx.logattacktime.mean', 'lowlevel.hfc.mean'] # modify according to features_and_keys                                   
                                             
for subdir, dirs, files in os.walk(files_dir):                          
    details = {}               
    dics = [json.load(open(subdir+'/'+f)) for f in files]
    details =  {'file': files, 'features': dics}

features = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(details['features']))     
files_features = np.array_split(features, len(files))  
      
zip_features_names = zip(*details['features'])
features_and_keys = [{'features':x, 'key':i} for i,x in enumerate(zip_features_names)]                
keys = [i for i,x in enumerate(features_and_keys)]

print "Before clustering, it is necessary to know which descriptors are going to be used. First descriptor will be at the x axis of the plot and Second descriptor will be at the y axis of the plot."
print (Fore.GREEN + "(Descriptors on the left and Keys on the right)")
print np.vstack((descriptors,keys)).T

first_descriptor_key = input("Key of first list of descriptors: ")                                                                      
second_descriptor_key = input("Key of second list of descriptors: ")
                                                            
if first_descriptor_key not in keys:                                                                     
	raise IndexError("Need keys of descriptors") 

if second_descriptor_key not in keys:                                                                     
	raise IndexError("Need keys of descriptors") 

print ("First descriptor is" + repr(descriptors[first_descriptor_key]))
print ("Second descriptor is" + repr(descriptors[second_descriptor_key]))

time.sleep(2)

print (Fore.MAGENTA + "Clustering")
         
first_descriptor_values = (zip(*files_features)[first_descriptor_key])                                                                  
second_descriptor_values = (zip(*files_features)[second_descriptor_key])            

min_to_max = preprocessing.MinMaxScaler(feature_range=(-1, 1))          
lowest_to_highest = min_to_max.fit_transform(np.vstack((first_descriptor_values,second_descriptor_values)).T)   
euclidean = AffinityPropagation(convergence_iter=10, affinity='euclidean')                           
euclidean_labels= euclidean.fit_predict(lowest_to_highest)

time.sleep(5)  

print (Fore.WHITE + "Cada número representa el grupo al que pertence el sonido como ejemplar de otro/s. El grupo '0' esta coloreado en azul, el grupo '1' esta coloreado en rojo, el grupo '2' esta coloreado en amarillo. Observa el ploteo para ver qué sonidos son ejemplares de otros")
print np.vstack((euclidean_labels,files)).T

time.sleep(6)

plt.scatter(lowest_to_highest[euclidean_labels==0,0], lowest_to_highest[euclidean_labels==0,1], c='b')
plt.scatter(lowest_to_highest[euclidean_labels==1,0], lowest_to_highest[euclidean_labels==1,1], c='r')
plt.scatter(lowest_to_highest[euclidean_labels==2,0], lowest_to_highest[euclidean_labels==2,1], c='y')
plt.xlabel(str(descriptors[first_descriptor_key])+"(scaled)")
plt.ylabel(str(descriptors[second_descriptor_key])+"(scaled)")
plt.show()
