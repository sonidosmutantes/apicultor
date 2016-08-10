import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                          
import json    
import re                                                         
from sklearn import preprocessing                                       
from sklearn.cluster import AffinityPropagation                                  
                                                                        
files_dir = 'descriptores/guitarra'                                            
                                             
for subdir, dirs, files in os.walk(files_dir):                          
    dic = [json.load(open(subdir+'/'+f)) for f in files]

features = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(dic))     
files_features = np.array_split(features, len(files))
min_to_max = preprocessing.MinMaxScaler(feature_range=(-1, 1))          
lowest_to_highest = min_to_max.fit_transform(files_features)  
euclidean = AffinityPropagation()                           
euclidean_labels = euclidean.fit_predict(lowest_to_highest)  

print files
print files_features
print euclidean_labels
