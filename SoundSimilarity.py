#! /usr/bin/env python
# -*- coding: utf-8 -*-


import time
from colorama import Fore
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                          
import json    
import re                                                         
from sklearn import preprocessing                                       
from sklearn.cluster import AffinityPropagation
import shutil                                  

def get_files(files_dir):
	"""
	return a list of .json files to read

	:param files_dir: directory where sounds are located
	"""
	files = [(files) for subdir, dirs, files in os.walk(files_dir+'/descriptores')]
	return files

def get_dics(files_dir):
	"""
	return a list containing all descriptions of tag directory reading the .json files

	:param files_dir: directory where sounds are located
	"""
	for subdir, dirs, files in os.walk(files_dir+'/descriptores'):
		dics = [json.load(open(subdir+'/'+f)) for f in files]  
	return dics

# plot sound similarity clusters
def plot_similarity_clusters(files, dics):
	"""
	find similar sounds using Affinity Propagation clusters

	:param files: the .json files (use get_files)
	:param dics: the list of descriptions (use get_dics)
	:returns:
	  - descriptors[first_descriptor_key], descriptors[second_descriptor_key]: the first descriptor used for clustering, the second descriptor used for clustering
	  - euclidean_labels: labels of clusters
	"""
	descriptors = ['lowlevel.dissonance.mean', 'lowlevel.mfcc_bands.mean', 'sfx.inharmonicity.mean', 'rhythm.bpm.mean', 'lowlevel.spectral_contrast.mean', 'lowlevel.spectral_centroid.mean', 'metadata.duration.mean', 'lowlevel.mfcc.mean', 'loudness.level.mean', 'rhythm.bpm_ticks.mean', 'lowlevel.spectral_valleys.mean', 'sfx.logattacktime.mean', 'lowlevel.hfc.mean'] # modify according to features_and_keys 

	features = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(dics))    
	files_features = np.array_split(features, np.array(files).size)  
	      
	zip_features_names = zip(*dics)
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
	plt.scatter(lowest_to_highest[euclidean_labels==3,0], lowest_to_highest[euclidean_labels==3,1], c='g')
	plt.xlabel(str(descriptors[first_descriptor_key])+"(scaled)")
	plt.ylabel(str(descriptors[second_descriptor_key])+"(scaled)")
	plt.show()

	return descriptors[first_descriptor_key], descriptors[second_descriptor_key], euclidean_labels


# save clusters files in clusters directory
def bpm_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """                                                                                     
    locate bpm according to clusters in clusters directories                                
                                                                                            
    :param files_dir: directory where sounds are located                                    
    :param descriptor: descriptor used for similarity                                       
    :param euclidean_labels: groups of clusters                                             
    :param files: the .json files (use get_files)                                           
    """                                                                                     
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]                       
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]
                                                                       
    for i, x in enumerate(euclidean_labels):                           
        if x == 2:                                                     
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)     
            files3 = [i.split('.json')[0] for i in group3]             
        else:                                                                                                                
            print ("Reading groups")                                                                                         
                                                                                                                             
    for i, x in enumerate(euclidean_labels):                                                                                 
        if x == 3:                                                                                                           
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]                                              
            group4 = map(lambda json: files[0][json], fourth_group)                                                          
            files4 = [i.split('.json')[0] for i in group4]                                                                   
        else:                                                                                                                
            print ("Reading groups")                                                                                         
                                                                                                                             
    print ("Saving files in clusters directory")                                                                             
                                                                                                                             
    group1 = map(lambda json: files[0][json], first_group)                                                                   
    group2 = map(lambda json: files[0][json], second_group) 
                                                           
    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]
                                                  
    for subdirs, dirs, sounds in os.walk(files_dir+'/tempo'):
        for s in list(sounds):                               
            sound_names = [s.split('tempo.wav')[0] for s in sounds]   
            break                                                     
                 
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:                                           
        files_3 = set(sound_names).intersection(files3)
        if files_3:                                    
                  os.mkdir(files_dir+'/tempo/2')
        for e in files_3:                          
                  shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.wav', files_dir+'/tempo/2/'+(str(e))+'tempo.wav')          
    except:                                                                                                                  
            print ("Creating two directories of clusters") 
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/tempo/0')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/tempo/1')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.wav', files_dir+'/tempo/0/'+(str(e))+'tempo.wav')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.wav', files_dir+'/tempo/1/'+(str(e))+'tempo.wav')

# save mfcc clusters files in mfcc clusters directory
def mfcc_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate mfcc files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups") 

    print ("Saving files in clusters directory")

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/mfcc'): 
        for s in list(sounds):
            sound_names = [s.split('mfcc.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/mfcc/2')
        for e in files_3:
            shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.wav', files_dir+'/mfcc/2/'+(str(e))+'mfcc.wav')
    except:
        print ("Creating two directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/mfcc/0')

    if files_2:
        os.mkdir(files_dir+'/mfcc/1')

    for e in files_1:
        shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.wav', files_dir+'/mfcc/0/'+(str(e))+'mfcc.wav')

    for e in files_2:
        shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.wav', files_dir+'/mfcc/1/'+(str(e))+'mfcc.wav')

# save contrast cluster files in contrast cluster directory
def contrast_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate contrast files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory")

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/valleys'): 
        for s in list(sounds):
            sound_names = [s.split('contrast.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/valleys/2')
        for e in files_3:
            shutil.copy(files_dir+'/valleys/'+(str(e))+'contrast.wav', files_dir+'/valleys/2/'+(str(e))+'contrast.wav')
    except:
        print ("Creating two directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/valleys/0')

    if files_2:
        os.mkdir(files_dir+'/valleys/1')

    for e in files_1:
        shutil.copy(files_dir+'/valleys/'+(str(e))+'contrast.wav', files_dir+'/valleys/0/'+(str(e))+'contrast.wav')

    for e in files_2:
        shutil.copy(files_dir+'/valleys/'+(str(e))+'contrast.wav', files_dir+'/valleys/1/'+(str(e))+'contrast.wav')

# save centroid cluster files in centroid cluster directory
def centroid_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate centroid files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory")

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/centroid'): 
        for s in list(sounds):
            sound_names = [s.split('centroid.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/centroid/2')
        for e in files_3:
            shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.wav', files_dir+'/centroid/2/'+(str(e))+'centroid.wav')
    except:
        print ("Creating two directories of clusters")

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/centroid/3')
        for e in files_4:
            shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.wav', files_dir+'/centroid/3/'+(str(e))+'centroid.wav')
    except:
        print ("Creating directories of clusters")  

    if files_1:
        os.mkdir(files_dir+'/centroid/0')

    if files_2:
        os.mkdir(files_dir+'/centroid/1')

    for e in files_1:
        shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.wav', files_dir+'/centroid/0/'+(str(e))+'centroid.wav')

    for e in files_2:
        shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.wav', files_dir+'/centroid/1/'+(str(e))+'centroid.wav')

# save loudness cluster files in loudness cluster directory
def loudness_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate loudness files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")


    print ("Saving files in clusters directory")

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/loudness'): 
        for s in list(sounds):
            sound_names = [s.split('loudness.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/loudness/2')
        for e in files_3:
            shutil.copy(files_dir+'/loudness/'+(str(e))+'loudness.wav', files_dir+'/loudness/2/'+(str(e))+'loudness.wav')
    except:
        print ("Creating two directories of clusters") 

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/loudness/3')
        for e in files_4:
            shutil.copy(files_dir+'/loudness/'+(str(e))+'loudness.wav', files_dir+'/loudness/3/'+(str(e))+'loudness.wav')
    except:
        print ("Creating directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/loudness/0')

    if files_2:
        os.mkdir(files_dir+'/loudness/1')

    for e in files_1:
        shutil.copy(files_dir+'/loudness/'+(str(e))+'loudness.wav', files_dir+'/loudness/0/'+(str(e))+'loudness.wav')

    for e in files_2:
        shutil.copy(files_dir+'/loudness/'+(str(e))+'loudness.wav', files_dir+'/loudness/1/'+(str(e))+'loudness.wav') 

# save hfc cluster files in hfc cluster directory
def hfc_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate hfc files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory") 

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/hfc'): 
        for s in list(sounds):
            sound_names = [s.split('hfc.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/hfc/2')
        for e in files_3:
            shutil.copy(files_dir+'/hfc/'+(str(e))+'hfc.wav', files_dir+'/hfc/2/'+(str(e))+'hfc.wav')
    except:
        print ("Creating two directories of clusters") 

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/hfc/3')
        for e in files_4:
            shutil.copy(files_dir+'/hfc/'+(str(e))+'hfc.wav', files_dir+'/hfc/3/'+(str(e))+'hfc.wav')
    except:
        print ("Creating directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/hfc/0')

    if files_2:
        os.mkdir(files_dir+'/hfc/1')

    for e in files_1:
        shutil.copy(files_dir+'/hfc/'+(str(e))+'hfc.wav', files_dir+'/hfc/0/'+(str(e))+'hfc.wav')

    for e in files_2:
        shutil.copy(files_dir+'/hfc/'+(str(e))+'hfc.wav', files_dir+'/hfc/1/'+(str(e))+'hfc.wav')


# save inharmonicity cluster files in inharmonicity cluster directory
def inharmonicity_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate inharmonicity files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory") 

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/inharmonicity'): 
        for s in list(sounds):
            sound_names = [s.split('inharmonicity.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/inharmonicity/2')
        for e in files_3:
            shutil.copy(files_dir+'/inharmonicity/'+(str(e))+'inharmonicity.wav', files_dir+'/inharmonicity/2/'+(str(e))+'inharmonicity.wav')
    except:
        print ("Creating two directories of clusters") 

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/inharmonicity/3')
        for e in files_4:
            shutil.copy(files_dir+'/inharmonicity/'+(str(e))+'inharmonicity.wav', files_dir+'/inharmonicity/3/'+(str(e))+'inharmonicity.wav')
    except:
        print ("Creating directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/inharmonicity/0')

    if files_2:
        os.mkdir(files_dir+'/inharmonicity/1')

    for e in files_1:
        shutil.copy(files_dir+'/inharmonicity/'+(str(e))+'inharmonicity.wav', files_dir+'/inharmonicity/0/'+(str(e))+'inharmonicity.wav')

    for e in files_2:
        shutil.copy(files_dir+'/inharmonicity/'+(str(e))+'inharmonicity.wav', files_dir+'/inharmonicity/1/'+(str(e))+'inharmonicity.wav')


# save dissonance cluster files in dissonance cluster directory
def dissonance_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate dissonance files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory") 

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/dissonance'): 
        for s in list(sounds):
            sound_names = [s.split('dissonance.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/dissonance/2')
        for e in files_3:
            shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.wav', files_dir+'/dissonance/2/'+(str(e))+'dissonance.wav')
    except:
        print ("Creating two directories of clusters") 

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/dissonance/3')
        for e in files_4:
            shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.wav', files_dir+'/dissonance/3/'+(str(e))+'dissonance.wav')
    except:
        print ("Creating directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/dissonance/0')

    if files_2:
        os.mkdir(files_dir+'/dissonance/1')

    for e in files_1:
        shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.wav', files_dir+'/dissonance/0/'+(str(e))+'dissonance.wav')

    for e in files_2:
        shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.wav', files_dir+'/dissonance/1/'+(str(e))+'dissonance.wav')


# save attack cluster files in attack cluster directory
def attack_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate attack files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory") 

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/attack'): 
        for s in list(sounds):
            sound_names = [s.split('attack.wav')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/attack/2')
        for e in files_3:
            shutil.copy(files_dir+'/attack/'+(str(e))+'attack.wav', files_dir+'/attack/2/'+(str(e))+'attack.wav')
    except:
        print ("Creating two directories of clusters") 

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/attack/3')
        for e in files_4:
            shutil.copy(files_dir+'/attack/'+(str(e))+'attack.wav', files_dir+'/attack/3/'+(str(e))+'attack.wav')
    except:
        print ("Creating directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/attack/0')

    if files_2:
        os.mkdir(files_dir+'/attack/1')

    for e in files_1:
        shutil.copy(files_dir+'/attack/'+(str(e))+'attack.wav', files_dir+'/attack/0/'+(str(e))+'attack.wav')

    for e in files_2:
        shutil.copy(files_dir+'/attack/'+(str(e))+'attack.wav', files_dir+'/attack/1/'+(str(e))+'attack.wav')
 

# save duration cluster files in dissonance cluster directory
def duration_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate duration files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    first_group = [i for i, x in enumerate(euclidean_labels) if x == 0]
    second_group = [i for i, x in enumerate(euclidean_labels) if x == 1]

    for i, x in enumerate(euclidean_labels):
        if x == 2:
            third_group = [i for i, x in enumerate(euclidean_labels) if x ==2]
            group3 = map(lambda json: files[0][json], third_group)
            files3 = [i.split('.json')[0] for i in group3]
        else:
            print ("Reading groups")

    for i, x in enumerate(euclidean_labels):
        if x == 3:
            fourth_group = [i for i, x in enumerate(euclidean_labels) if x ==3]
            group4 = map(lambda json: files[0][json], fourth_group)
            files4 = [i.split('.json')[0] for i in group4]
        else:
            print ("Reading groups")

    print ("Saving files in clusters directory") 

    group1 = map(lambda json: files[0][json], first_group)
    group2 = map(lambda json: files[0][json], second_group)

    files1 = [i.split('.json')[0] for i in group1]
    files2 = [i.split('.json')[0] for i in group2]

    for subdirs, dirs, sounds in os.walk(files_dir+'/duration'): 
        for s in list(sounds):
            sound_names = [s.split('.ogg')[0] for s in sounds]
            break
            
    files_1 = set(sound_names).intersection(files1)
    files_2 = set(sound_names).intersection(files2)
    try:
        files_3 = set(sound_names).intersection(files3)
        if files_3:
            os.mkdir(files_dir+'/duration/2')
        for e in files_3:
            shutil.copy(files_dir+'/duration/'+(str(e))+'.ogg', files_dir+'/duration/2/'+(str(e))+'duration.ogg')
    except:
        print ("Creating two directories of clusters") 

    try:
        files_4 = set(sound_names).intersection(files4)
        if files_4:
            os.mkdir(files_dir+'/duration/3')
        for e in files_4:
            shutil.copy(files_dir+'/duration/'+(str(e))+'.ogg', files_dir+'/duration/3/'+(str(e))+'duration.ogg')
    except:
        print ("Creating directories of clusters") 

    if files_1:
        os.mkdir(files_dir+'/duration/0')

    if files_2:
        os.mkdir(files_dir+'/duration/1')

    for e in files_1:
        shutil.copy(files_dir+'/duration/'+(str(e))+'.ogg', files_dir+'/duration/0/'+(str(e))+'duration.ogg')

    for e in files_2:
        shutil.copy(files_dir+'/duration/'+(str(e))+'.ogg', files_dir+'/duration/1/'+(str(e))+'duration.ogg')


Usage = "./SoundSimilarity.py [FILES_DIR]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        files_dir = sys.argv[1]

    	if not os.path.exists(files_dir+'/descriptores'):                         
		raise IOError("Must run MIR analysis or set the sounds/tag directory as files_dir to find descriptors directory")

	files = get_files(files_dir)
	dics = get_dics(files_dir)
	desc1, desc2, euclidean_labels = plot_similarity_clusters(files, dics)

	time.sleep(2)

	print ("If you chose both rhythm descriptors, both contrast descriptors or both mfcc descriptors clusters of sounds will be located in the same directory")

	if 'rhythm.bpm.mean'not in desc1:
	    print ("First Descriptor is not rhythm.bpm.mean")
	else:   
	    try:   
		bpm_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'rhythm.bpm.mean'not in desc2:
	    print ("Second Descriptor is not rhythm.bpm.mean")
	else:
	    try:   
		bpm_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'rhythm.bpm_ticks.mean'not in desc1:
	    print ("First Descriptor is not rhythm.bpm_ticks.mean")
	else:
	    try:   
		bpm_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'rhythm.bpm.mean'not in desc2:
	    print ("Second Descriptor is not rhythm.bpm_ticks.mean")
	else:   
	    try:
		bpm_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mffc_bands.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.mfcc_bands.mean")
	else:   
	    try:   
		mfcc_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mfcc_bands.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.mfcc_bands.mean")
	else:
	    try:   
		mfcc_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mffc.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.mfcc.mean")
	else:   
	    try:   
		mfcc_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mfcc.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.mfcc.mean")
	else:
	    try:   
		mfcc_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_contrast.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.spectral_contrast.mean")
	else:   
	    try:   
		contrast_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.spectral_contrast.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.spectral_contrast.mean")
	else:
	    try:   
		contrast_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_valleys.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.spectral_valleys.mean")
	else:
	    try:   
		contrast_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_valleys.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.spectral_valleys.mean")
	else:   
	    try:
		contrast_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_centroid.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.spectral_centroid.mean")
	else:   
	    try:   
		centroid_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.spectral_centroid.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.spectral_centroid.mean")
	else:
	    try:   
		centroid_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'loudness.level.mean'not in desc1:
	    print ("First Descriptor is not loudness.level.mean")
	else:   
	    try:   
		loudness_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'loudness.level.mean'not in desc2:
	    print ("Second Descriptor is not loudness.level.mean")
	else:
	    try:   
		loudness_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.hfc.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.hfc.mean")
	else:   
	    try:   
		contrast_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.hfc.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.hfc.mean")
	else:
	    try:   
		hfc_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'sfx.inharmonicity.mean'not in desc1:
	    print ("First Descriptor is not sfx.inharmonicity.mean")
	else:   
	    try:   
		inharmonicity_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'sfx.inharmonicity.mean'not in desc2:
	    print ("Second Descriptor is not sfx.inharmonicity.mean")
	else:
	    try:   
		inharmonicity_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'lowlevel.dissonance.mean'not in desc1:
	    print ("First Descriptor is not lowlevel.dissonance.mean")
	else:   
	    try:   
		dissonance_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.dissonance.mean'not in desc2:
	    print ("Second Descriptor is not lowlevel.dissonance.mean")
	else:
	    try:   
		dissonance_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'sfx.logattacktime.mean'not in desc1:
	    print ("First Descriptor is not sfx.logattacktime.mean")
	else:   
	    try:   
		attack_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'sfx.logattacktime.mean'not in desc2:
	    print ("Second Descriptor is not sfx.logattacktime.mean")
	else:
	    try:   
		attack_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'metadata.duration.mean'not in desc1:
	    print ("First Descriptor is not metadata.duration.mean")
	else:   
	    try:   
		duration_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'metadata.duration.mean'not in desc2:
	    print ("Second Descriptor is not metadata.duration.mean")
	else:
	    try:   
		duration_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	print ("Saved clusters of selected descriptors")

    except Exception, e:
        print(e)
        exit(1) 
