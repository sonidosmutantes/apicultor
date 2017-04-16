#! /usr/bin/env python
# -*- coding: utf-8 -*-

from MusicEmotionMachine import scratch_music
from utils.data import *
import time
from colorama import Fore
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                      
from sklearn import preprocessing
from essentia.standard import *
from sklearn.decomposition.pca import PCA                                      
from sklearn.cluster import AffinityPropagation
from collections import defaultdict
from random import choice
import shutil
import librosa
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_desc_pair(descriptors, files_features, keys):

	print np.vstack((descriptors,keys)).T

	input1 = input("Key of first list of descriptors:")                                                                      
	input2 = input("Key of second list of descriptors: ")
				                                            
	if input1 not in keys:                                                                     
		raise IndexError("Need keys of descriptors") 

	if input2 not in keys:                                                                     
		raise IndexError("Need keys of descriptors") 
			 
	first_descriptor_values = (zip(*files_features)[input1])                                                                  
	second_descriptor_values = (zip(*files_features)[input2])  

	return first_descriptor_values, second_descriptor_values, descriptors[input1], descriptors[input2]


# plot sound similarity clusters
def plot_similarity_clusters(desc1, desc2, plot = None):
	"""
	find similar sounds using Affinity Propagation clusters

	:param desc1: first descriptor values
	:param desc2: second descriptor values
	:returns:
	  - euclidean_labels: labels of clusters
	""" 

	if plot == True:
		print (Fore.MAGENTA + "Clustering")
	else:
		pass
         
	min_max = preprocessing.scale(np.vstack((desc1,desc2)).T, with_mean=False, with_std=False)          
	pca = PCA(n_components=2, whiten=True)
	y = pca.fit(min_max).transform(min_max)
	    
	euclidean = AffinityPropagation(convergence_iter=1800, affinity='euclidean')                           
	euclidean_labels= euclidean.fit_predict(y)

	if plot == True:

		time.sleep(5)  

		print (Fore.WHITE + "Cada número representa el grupo al que pertence el sonido como ejemplar de otro/s. El grupo '0' esta coloreado en azul, el grupo '1' esta coloreado en rojo, el grupo '2' esta coloreado en amarillo. Observa el ploteo para ver qué sonidos son ejemplares de otros")
		print np.vstack((euclidean_labels,files)).T

		time.sleep(6)

		plt.scatter(y[euclidean_labels==0,0], y[euclidean_labels==0,1], c='b')
		plt.scatter(y[euclidean_labels==1,0], y[euclidean_labels==1,1], c='r')
		plt.scatter(y[euclidean_labels==2,0], y[euclidean_labels==2,1], c='y')
		plt.scatter(y[euclidean_labels==3,0], y[euclidean_labels==3,1], c='g')
		plt.show()
	else:
		pass

	return euclidean_labels


# save clusters files in clusters directory
def bpm_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """                                                                                     
    locate bpm according to clusters in clusters directories                                
                                                                                            
    :param files_dir: directory where sounds are located                                    
    :param descriptor: descriptor used for similarity                                       
    :param euclidean_labels: groups of clusters                                             
    :param files: the .json files (use get_files)                                           
    """  
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/tempo/'+str(c)):
            os.makedirs(files_dir+'/tempo/'+str(c))
            os.makedirs(files_dir+'/tempo/'+str(c)+'/remix')
        for t in groups[c]:
            for s in list(os.walk(files_dir+'/tempo', topdown=False))[-1][-1]:
                 if str(t[0]).split('.json')[0] == s.split('tempo.ogg')[0]:
                     shutil.copy(files_dir+'/tempo/'+s, files_dir+'/tempo/'+str(c)+'/'+s)     
                     print str().join((str(t))) 
        try:
            simil_audio = [MonoLoader(filename=files_dir+'/tempo/'+str(c)+f)() for f in list(os.walk(files_dir+'/tempo/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                               
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                               
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, h
            p = librosa.istft(p)                                                
            MonoWriter(filename=files_dir+'/tempo/'+str(c)+'/remix/'+'similarity_mix_bpm.ogg', format = 'ogg', sampleRate = 44100)(p)  
            del p
        except Exception, e:
            print logger.exception(e)
            continue

# save mfcc clusters files in mfcc clusters directory
def mfcc_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate mfcc files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/mfcc/'+str(c)):
            os.makedirs(files_dir+'/mfcc/'+str(c))
            os.makedirs(files_dir+'/mfcc/'+str(c)+'/remix')            
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/mfcc', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('mfcc.ogg')[0]:
                     shutil.copy(files_dir+'/mfcc/'+s, files_dir+'/mfcc/'+str(c)+'/'+s)     
                     print t
 
        try: 
            simil_audio = [MonoLoader(filename=files_dir+'/mfcc/'+str(c)+f)() for f in list(os.walk(files_dir+'/mfcc/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                   
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/mfcc/'+str(c)+'/remix/'+'similarity_mix_mfcc.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue

# save contrast cluster files in contrast cluster directory
def contrast_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate contrast files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/valleys/'+str(c)):
            os.makedirs(files_dir+'/valleys/'+str(c))
            os.makedirs(files_dir+'/valleys/'+str(c)+'/remix')  
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/valleys', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('valleys.ogg')[0]:
                     shutil.copy(files_dir+'/valleys/'+s, files_dir+'/valleys/'+str(c)+'/'+s)     
                     print t 
 
        try:
            simil_audio = [MonoLoader(filename=files_dir+'/valleys/'+str(c)+f)() for f in list(os.walk(files_dir+'/valleys/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                               
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                  
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/valleys/'+str(c)+'/remix/'+'similarity_mix_valleys.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue
            
# save centroid cluster files in centroid cluster directory
def centroid_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate centroid files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/centroid/'+str(c)):
            os.makedirs(files_dir+'/centroid/'+str(c))
            os.makedirs(files_dir+'/centroid/'+str(c)+'/remix')        
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/centroid', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('centroid.ogg')[0]:
                     shutil.copy(files_dir+'/centroid/'+s, files_dir+'/centroid/'+str(c)+'/'+s)     
                     print t 
 
        try: 
            simil_audio = [MonoLoader(filename=files_dir+'/centroid/'+str(c)+f)() for f in list(os.walk(files_dir+'/centroid/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                               
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                  
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/centroid/'+str(c)+'/remix/'+'similarity_mix_centroid.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue
            
# save loudness cluster files in loudness cluster directory
def loudness_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate loudness files according to clusters in clusters directories
    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/loudness/'+str(c)):
            os.makedirs(files_dir+'/loudness/'+str(c))
            os.makedirs(files_dir+'/loudness/'+str(c)+'/remix')    
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/loudness', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('loudness.ogg')[0]:
                     shutil.copy(files_dir+'/loudness/'+s, files_dir+'/loudness/'+str(c)+'/'+s)     
                     print t 
 
        try: 
            simil_audio = [MonoLoader(filename=files_dir+'/loudness/'+str(c)+f)() for f in list(os.walk(files_dir+'/loudness/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio))
            del simil_audio                                
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                  
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/loudness/'+str(c)+'/remix/'+'similarity_mix_centroid.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue

# save hfc cluster files in hfc cluster directory
def hfc_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate hfc files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/hfc/'+str(c)):
            os.makedirs(files_dir+'/hfc/'+str(c))
            os.makedirs(files_dir+'/hfc/'+str(c)+'/remix')   
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/hfc', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('hfc.ogg')[0]:
                     shutil.copy(files_dir+'/hfc/'+s, files_dir+'/hfc/'+str(c)+'/'+s)     
                     print t 
 
        try:  
            simil_audio = [MonoLoader(filename=files_dir+'/hfc/'+str(c)+f)() for f in list(os.walk(files_dir+'/hfc/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio))
            del simil_audio                                
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                  
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, h
            p = librosa.istft(p)                                                
            MonoWriter(filename=files_dir+'/hfc/'+str(c)+'/remix/'+'similarity_mix_hfc.ogg', format = 'ogg', sampleRate = 44100)(p)  
            del p
        except Exception, e:
            print e
            continue
            
# save inharmonicity cluster files in inharmonicity cluster directory
def inharmonicity_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate inharmonicity files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/inharmonicity/'+str(c)):
            os.makedirs(files_dir+'/inharmonicity/'+str(c))
            os.makedirs(files_dir+'/inharmonicity/'+str(c)+'/remix')   
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/inharmonicity', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('inharmonicity.ogg')[0]:
                     shutil.copy(files_dir+'/inharmonicity/'+s, files_dir+'/inharmonicity/'+str(c)+'/'+s)     
                     print t 
 
        try:
            simil_audio = [MonoLoader(filename=files_dir+'/inharmonicity/'+str(c)+f)() for f in list(os.walk(files_dir+'/inharmonicity/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                               
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                         
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/inharmonicity/'+str(c)+'/remix/'+'similarity_mix_inharmonicity.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue

# save dissonance cluster files in dissonance cluster directory
def dissonance_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate dissonance files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/dissonance/'+str(c)):
            os.makedirs(files_dir+'/dissonance/'+str(c))
            os.makedirs(files_dir+'/dissonance/'+str(c)+'/remix')   
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/dissonance', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('dissonance.ogg')[0]:
                     shutil.copy(files_dir+'/dissonance/'+s, files_dir+'/dissonance/'+str(c)+'/'+s)     
                     print t
 
        try: 
            simil_audio = [MonoLoader(filename=files_dir+'/dissonance/'+str(c)+f)() for f in list(os.walk(files_dir+'/dissonance/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                               
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                  
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples            
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p            
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/dissonance/'+str(c)+'/remix/'+'similarity_mix_dissonance.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue
            
# save attack cluster files in attack cluster directory
def attack_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate attack files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/attack/'+str(c)):
            os.makedirs(files_dir+'/attack/'+str(c))
            os.makedirs(files_dir+'/attack/'+str(c)+'/remix')    
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/attack', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('attack.ogg')[0]:
                     shutil.copy(files_dir+'/attack/'+s, files_dir+'/attack/'+str(c)+'/'+s)     
                     print t
 
        try:
            simil_audio = [MonoLoader(filename=files_dir+'/attack/'+str(c)+f)() for f in list(os.walk(files_dir+'/attack/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio)) 
            del simil_audio                               
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]]                                                  
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples      
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, h          
            p = librosa.istft(p)                                                
            MonoWriter(filename=files_dir+'/attack/'+str(c)+'/remix/'+'similarity_mix_attack.ogg', format = 'ogg', sampleRate = 44100)(p)  
            del p
        except Exception, e:
            print e
            continue
            
# save duration cluster files in dissonance cluster directory
def duration_cluster_files(files_dir, descriptor, euclidean_labels, files):
    """
    locate duration files according to clusters in clusters directories

    :param files_dir: directory where sounds are located
    :param descriptor: descriptor used for similarity
    :param euclidean_labels: groups of clusters 
    :param files: the .json files (use get_files)
    """
    groups = [[] for i in xrange(len(np.unique(euclidean_labels)))]
    for i, x in enumerate(euclidean_labels):
        groups[x].append([files[0][i], x])
                
    for c in xrange(len(groups)):
        if not os.path.exists(files_dir+'/duration/'+str(c)):
            os.makedirs(files_dir+'/duration/'+str(c))
            os.makedirs(files_dir+'/duration/'+str(c)+'/remix')     
        for t in groups[c]:       
            for s in list(os.walk(files_dir+'/duration', topdown=False))[-1][-1]:
                 if str(t[0]).split('.')[0] == s.split('duration.ogg')[0]:
                     shutil.copy(files_dir+'/duration/'+s, files_dir+'/duration/'+str(c)+'/'+s)     
                     print t
 
        try:
            simil_audio = [MonoLoader(filename=files_dir+'/duration/'+str(c)+f)() for f in list(os.walk(files_dir+'/duration/'+str(c), topdown = False))[-1][-1]]
            audio0 = scratch_music(choice(simil_audio))
            audio1 = scratch_music(choice(simil_audio))
            del simil_audio                                
            audio_N = min([len(i) for i in [audio0, audio1]])  
            audio_samples = [i[:audio_N]/i.max() for i in [audio0, audio1]] 
            del audio0, audio1                                                 
            simil_x = np.array(audio_samples).sum(axis=0) 
            del audio_samples           
            simil_x = 0.5*simil_x/simil_x.max()      
            h, p = librosa.decompose.hpss(librosa.core.stft(simil_x))
            del simil_x, p           
            h = librosa.istft(h)                                                
            MonoWriter(filename=files_dir+'/duration/'+str(c)+'/remix/'+'similarity_mix_duration.ogg', format = 'ogg', sampleRate = 44100)(h)  
            del h
        except Exception, e:
            print e
            continue
            
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
	descriptors = desc_pair(files,dics).descriptors
	files_features = desc_pair(files,dics).files_features
	keys = desc_pair(files,dics).keys
	desc1, desc2, in1, in2 = get_desc_pair(descriptors, files_features, keys)
	euclidean_labels = plot_similarity_clusters(desc1, desc2, plot = True)

	time.sleep(2)

	print ("If you chose both rhythm descriptors, both contrast descriptors or both mfcc descriptors clusters of sounds will be located in the same directory")

	if 'rhythm.bpm.mean'not in in1:
	    print ("First Descriptor is not rhythm.bpm.mean")
	else:   
	    try:   
		bpm_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'rhythm.bpm.mean'not in in2:
	    print ("Second Descriptor is not rhythm.bpm.mean")
	else:
	    try:   
		bpm_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'rhythm.bpm_ticks.mean'not in in1:
	    print ("First Descriptor is not rhythm.bpm_ticks.mean")
	else:
	    try:   
		bpm_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'rhythm.bpm.mean'not in in2:
	    print ("Second Descriptor is not rhythm.bpm_ticks.mean")
	else:   
	    try:
		bpm_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mffc_bands.mean'not in in1:
	    print ("First Descriptor is not lowlevel.mfcc_bands.mean")
	else:   
	    try:   
		mfcc_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mfcc_bands.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.mfcc_bands.mean")
	else:
	    try:   
		mfcc_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mffc.mean'not in in1:
	    print ("First Descriptor is not lowlevel.mfcc.mean")
	else:   
	    try:   
		mfcc_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.mfcc.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.mfcc.mean")
	else:
	    try:   
		mfcc_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_contrast.mean'not in in1:
	    print ("First Descriptor is not lowlevel.spectral_contrast.mean")
	else:   
	    try:   
		contrast_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.spectral_contrast.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.spectral_contrast.mean")
	else:
	    try:   
		contrast_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_valleys.mean'not in in1:
	    print ("First Descriptor is not lowlevel.spectral_valleys.mean")
	else:
	    try:   
		contrast_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_valleys.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.spectral_valleys.mean")
	else:   
	    try:
		contrast_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.spectral_centroid.mean'not in in1:
	    print ("First Descriptor is not lowlevel.spectral_centroid.mean")
	else:   
	    try:   
		centroid_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.spectral_centroid.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.spectral_centroid.mean")
	else:
	    try:   
		centroid_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'loudness.level.mean'not in in1:
	    print ("First Descriptor is not loudness.level.mean")
	else:   
	    try:   
		loudness_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'loudness.level.mean'not in in2:
	    print ("Second Descriptor is not loudness.level.mean")
	else:
	    try:   
		loudness_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")

	if 'lowlevel.hfc.mean'not in in1:
	    print ("First Descriptor is not lowlevel.hfc.mean")
	else:   
	    try:   
		contrast_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.hfc.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.hfc.mean")
	else:
	    try:   
		hfc_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'sfx.inharmonicity.mean'not in in1:
	    print ("First Descriptor is not sfx.inharmonicity.mean")
	else:   
	    try:   
		inharmonicity_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'sfx.inharmonicity.mean'not in in2:
	    print ("Second Descriptor is not sfx.inharmonicity.mean")
	else:
	    try:   
		inharmonicity_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'lowlevel.dissonance.mean'not in in1:
	    print ("First Descriptor is not lowlevel.dissonance.mean")
	else:   
	    try:   
		dissonance_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'lowlevel.dissonance.mean'not in in2:
	    print ("Second Descriptor is not lowlevel.dissonance.mean")
	else:
	    try:   
		dissonance_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'sfx.logattacktime.mean'not in in1:
	    print ("First Descriptor is not sfx.logattacktime.mean")
	else:   
	    try:   
		attack_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'sfx.logattacktime.mean'not in in2:
	    print ("Second Descriptor is not sfx.logattacktime.mean")
	else:
	    try:   
		attack_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	if 'metadata.duration.mean'not in in1:
	    print ("First Descriptor is not metadata.duration.mean")
	else:   
	    try:   
		duration_cluster_files(files_dir, desc1, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved")


	if 'metadata.duration.mean'not in in2:
	    print ("Second Descriptor is not metadata.duration.mean")
	else:
	    try:   
		duration_cluster_files(files_dir, desc2, euclidean_labels, files)
	    except OSError:
		print ("Error: Clusters already saved") 

	print ("Saved clusters of selected descriptors")

    except Exception, e:
        print logger.exception(e)
        sys.exit(1) 
