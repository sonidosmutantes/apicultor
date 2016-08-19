#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                          
import json    
import re                                                         
from sklearn import preprocessing                                       
from sklearn.cluster import AffinityPropagation                                  
                                                                        
files_dir = 'descriptores/teclado'                                            
                                             
for subdir, dirs, files in os.walk(files_dir):                          
    details = {}               
    dics = [json.load(open(subdir+'/'+f)) for f in files]
    details =  {'file': files, 'features': dics}

features = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(details['features']))     
files_features = np.array_split(features, len(files))  

first_descriptor_values = (zip(*files_features)[4])
second_descriptor_values = (zip(*files_features)[2])

min_to_max = preprocessing.MinMaxScaler(feature_range=(-1, 1))          
lowest_to_highest = min_to_max.fit_transform(np.vstack((first_descriptor_values,second_descriptor_values)).T)   
euclidean = AffinityPropagation(convergence_iter=10, affinity='euclidean')                           
euclidean_labels= euclidean.fit_predict(lowest_to_highest)  

print "Cada número representa grupos de sonidos ejemplares, observa el ploteo para ver qué sonidos podrían ser ejemplares de otros"
print np.vstack((euclidean_labels,files)).T
print lowest_to_highest

plt.scatter(lowest_to_highest[euclidean_labels==0,0], lowest_to_highest[euclidean_labels==0,1], c='b')
plt.scatter(lowest_to_highest[euclidean_labels==1,0], lowest_to_highest[euclidean_labels==1,1], c='r')
plt.scatter(lowest_to_highest[euclidean_labels==2,0], lowest_to_highest[euclidean_labels==2,1], c='y')
plt.xlabel('Spectral_Centroid (scaled)')
plt.ylabel('BPM (scaled)')
plt.show()
