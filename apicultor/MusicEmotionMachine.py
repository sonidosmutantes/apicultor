#! /usr/bin/env python
# -*- coding: utf-8 -*-

from SoundSimilarity import *
import time
from colorama import Fore
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                           
from essentia.standard import *
from smst.utils.audio import write_wav    
from sklearn import svm
import librosa
from librosa import *                                       
import shutil
from smst.models import stft 
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)                               

#emotion classification
def plot_emotion_clusters(files_dir, multitag = None):
	"""
	classify sounds based on emotivity (emotive or non-emotive)using Affinity Propagation labels

	:param files_dir: data tag dir if not performing multitag classification. data dir if performing multitag classification
	:param multitag: if True, will classify all downloaded files and remix when performing emotional state transition
	:returns:
	  - negative_emotion_files: files with negative emotional value (emotional meaning according to the whole performance)
	  - positive_emotion_files: files with positive emotional value (emotional meaning according to the whole performance) 
	"""
	if multitag == None:
		files = get_files(files_dir)
		dics = get_dics(files_dir)
	elif multitag == True:
		files = np.hstack([get_files(tag) for tag in files_dir])
		dics = np.hstack([get_dics(tag) for tag in files_dir])
	else:
		if multitag == False:
			pass

	print ("Selecting 'metadata.duration.mean' is not meaningful if you want to use the data for emotions classification")

	similar = plot_similarity_clusters(files, dics)
	labels = similar[2]
	y = similar[4]
	z_mean = labels-np.mean(labels) 
	labels[z_mean<0] = 0 
	labels[z_mean>0] = 1  
	clf = svm.SVC(kernel = 'poly', decision_function_shape = 'ovr', gamma = 2, class_weight = 'balanced').fit(y, labels)
	h = plot_similarity_clusters(files, dics)[4]
	l = plot_similarity_clusters(files, dics)[4]
	v = plot_similarity_clusters(files, dics)[4]
	x = plot_similarity_clusters(files, dics)[4]
	z = plot_similarity_clusters(files, dics)[4]
	xyz = np.sum([y, h, l, v, x, z], axis=0)
	labels = clf.predict(xyz)
	clf = svm.SVC(kernel = 'poly', decision_function_shape = 'ovr', gamma = 2, class_weight = 'balanced').fit(xyz, labels)

	x_min, x_max = xyz[:, 0].min() - 1, xyz[:, 0].max() + 1 
	y_min, y_max = xyz[:, 1].min() - 1, xyz[:, 1].max() + 1 
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape) 
 
	plt.contour(xx, yy, Z) 
	plt.scatter(xyz[:, 0], xyz[:, 1], c = labels)  

	print (Fore.WHITE + "El grupo negativo '0' esta coloreado en azul, el grupo positivo '1' esta coloreado en rojo")
	print np.vstack((labels,files)).T

	time.sleep(6)

	plt.show()

	negative_emotion_group = map(lambda json: files[0][json], [i for i, x in enumerate(labels) if x ==0])
	negative_emotion_files = [i.split('.json')[0] for i in negative_emotion_group]

	positive_emotion_group = map(lambda json: files[0][json], [i for i, x in enumerate(labels) if x ==1])
	positive_emotion_files = [i.split('.json')[0] for i in positive_emotion_group]

	return negative_emotion_files, positive_emotion_files

#create emotions directory in data dir if multitag classification has been performed
def emotions_data_dir():
    """                                                                                     
    create emotions directory for all data                                          
    """           
    files_dir = 'data'                                                     
    if not os.path.exists(files_dir+'/emotions/happy'):
        os.makedirs(files_dir+'/emotions/happy')
    if not os.path.exists(files_dir+'/emotions/sad'):
        os.makedirs(files_dir+'/emotions/sad')
    if not os.path.exists(files_dir+'/emotions/angry'):
        os.makedirs(files_dir+'/emotions/angry')
    if not os.path.exists(files_dir+'/emotions/relaxed'):
        os.makedirs(files_dir+'/emotions/relaxed')
    if not os.path.exists(files_dir+'/emotions/not happy'):
        os.makedirs(files_dir+'/emotions/not happy')
    if not os.path.exists(files_dir+'/emotions/not sad'):
        os.makedirs(files_dir+'/emotions/not sad')
    if not os.path.exists(files_dir+'/emotions/not angry'):
        os.makedirs(files_dir+'/emotions/not angry')
    if not os.path.exists(files_dir+'/emotions/not relaxed'):
        os.makedirs(files_dir+'/emotions/not relaxed')

#look for all downloaded audio
tags_dirs = [os.path.join('data',dirs) for dirs in next(os.walk(os.path.abspath('data')))[1]]

#classify all downloaded audio in tags
def multitag_emotion_classifier(tags_dirs):
    """                                                                                     
    emotion classification of all data                                
                                                                                            
    :param tags_dirs = paths of tags in data                                                                              
    """           
    neg_and_pos = plot_emotion_clusters(tags_dirs, multitag = True)   
    return neg_and_pos

#emotions dictionary directory (to use with RedPanal API)
def multitag_emotions_dictionary_dir():
    """                                                                                     
    create emotions dictionary directory                                        
    """           
    os.makedirs('data/emotions_dictionary')

# save files in tag directory according to emotion using hpss
def bpm_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is happy")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is sad")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(negative_emotion_files)
    files_2 = set(sound_names).intersection(positive_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/tempo/sad')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/tempo/happy')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.ogg', files_dir+'/tempo/sad/'+(str(e))+'tempo.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.ogg', files_dir+'/tempo/happy/'+(str(e))+'tempo.ogg')


    happiness_dir = files_dir+'/tempo/happy' 
    for subdirs, dirs, sounds in os.walk(happiness_dir):  
    	happy_audio = [MonoLoader(filename=happiness_dir+'/'+happy_f)() for happy_f in sounds]
    happy_N = min([len(i) for i in happy_audio])  
    happy_samples = [i[:happy_N]/i.max() for i in happy_audio]  
    happy_x = np.array(happy_samples).sum(axis=0) 
    happy_X = 0.5*happy_x/happy_x.max()
    happy_Harmonic, happy_Percussive = decompose.hpss(librosa.core.stft(happy_X))
    happy_harmonic = istft(happy_Harmonic) 
    MonoWriter(filename=files_dir+'/tempo/happy/'+'happy_mix_bpm.ogg', format = 'ogg', sampleRate = 44100)(happy_harmonic)  

    sadness_dir = files_dir+'/tempo/sad'
    for subdirs, dirs, sad_sounds in os.walk(sadness_dir):
    	sad_audio = [MonoLoader(filename=sadness_dir+'/'+sad_f)() for sad_f in sad_sounds]
    sad_N = min([len(i) for i in sad_audio])  
    sad_samples = [i[:sad_N]/i.max() for i in sad_audio]  
    sad_x = np.array(sad_samples).sum(axis=0) 
    sad_X = 0.5*sad_x/sad_x.max()
    sad_Harmonic, sad_Percussive = decompose.hpss(librosa.core.stft(sad_X))
    sad_harmonic = istft(sad_Harmonic)  
    MonoWriter(filename=files_dir+'/tempo/sad/'+'sad_mix_bpm.ogg', format = 'ogg', sampleRate = 44100)(sad_harmonic) 

def attack_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is angry")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/attack/angry')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/attack/relaxed')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/attack/'+(str(e))+'attack.ogg', files_dir+'/attack/angry/'+(str(e))+'attack.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/attack/'+(str(e))+'attack.ogg', files_dir+'/attack/relaxed/'+(str(e))+'attack.ogg')


    anger_dir = files_dir+'/attack/angry' 
    for subdirs, dirs, sounds in os.walk(anger_dir):  
    	angry_audio = [MonoLoader(filename=anger_dir+'/'+angry_f)() for angry_f in sounds]
    angry_N = min([len(i) for i in angry_audio])  
    angry_samples = [i[:angry_N]/i.max() for i in angry_audio]  
    angry_x = np.array(angry_samples).sum(axis=0) 
    angry_X = 0.5*angry_x/angry_x.max()
    angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(angry_X))
    angry_harmonic = istft(angry_Harmonic) 
    MonoWriter(filename=files_dir+'/attack/angry/angry_mix_attack.ogg', format = 'ogg', sampleRate = 44100)(angry_harmonic)

    tenderness_dir = files_dir+'/attack/relaxed'
    for subdirs, dirs, tender_sounds in os.walk(tenderness_dir):
    	tender_audio = [MonoLoader(filename=tenderness_dir+'/'+tender_f)() for tender_f in tender_sounds]
    tender_N = min([len(i) for i in tender_audio])  
    tender_samples = [i[:tender_N]/i.max() for i in tender_audio]  
    tender_x = np.array(tender_samples).sum(axis=0) 
    tender_X = 0.5*tender_x/tender_x.max()
    tender_Harmonic, tender_Percussive = decompose.hpss(librosa.core.stft(tender_X))
    tender_harmonic = istft(tender_Harmonic)  
    MonoWriter(filename=files_dir+'/attack/relaxed/relaxed_mix_attack.ogg', format = 'ogg', sampleRate = 44100)(tender_harmonic) 

def dissonance_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is angry")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/dissonance/angry')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/dissonance/relaxed')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.ogg', files_dir+'/dissonance/angry/'+(str(e))+'dissonance.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.ogg', files_dir+'/dissonance/relaxed/'+(str(e))+'dissonance.ogg')


    fear_dir = files_dir+'/dissonance/angry' 
    for subdirs, dirs, sounds in os.walk(fear_dir):  
    	fear_audio = [MonoLoader(filename=fear_dir+'/'+fear_f)() for fear_f in sounds]
    fear_N = min([len(i) for i in fear_audio])  
    fear_samples = [i[:fear_N]/i.max() for i in fear_audio]  
    fear_x = np.array(fear_samples).sum(axis=0) 
    fear_X = 0.5*fear_x/fear_x.max()
    fear_Harmonic, fear_Percussive = decompose.hpss(librosa.core.stft(fear_X))
    fear_harmonic = istft(fear_Harmonic) 
    MonoWriter(filename=files_dir+'/dissonance/angry/angry_mix_dissonance.ogg', format = 'ogg', sampleRate = 44100)(fear_harmonic)
  

    happiness_dir = files_dir+'/dissonance/relaxed'
    for subdirs, dirs, happy_sounds in os.walk(happiness_dir):
    	happy_audio = [MonoLoader(filename=happiness_dir+'/'+happy_f)() for happy_f in happy_sounds]
    happy_N = min([len(i) for i in happy_audio])  
    happy_samples = [i[:happy_N]/i.max() for i in happy_audio]  
    happy_x = np.array(happy_samples).sum(axis=0) 
    happy_X = 0.5*happy_x/happy_x.max()
    happy_Harmonic, happy_Percussive = decompose.hpss(librosa.core.stft(happy_X))
    happy_harmonic = istft(happy_Harmonic)  
    MonoWriter(filename=files_dir+'/dissonance/relaxed/relaxed_mix_dissonance.ogg', format = 'ogg', sampleRate = 44100)(happy_harmonic) 

def mfcc_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is angry")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/mfcc/angry')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/mfcc/relaxed')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.ogg', files_dir+'/mfcc/angry/'+(str(e))+'mfcc.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.ogg', files_dir+'/mfcc/relaxed/'+(str(e))+'mfcc.ogg')


    fear_dir = files_dir+'/mfcc/angry' 
    for subdirs, dirs, sounds in os.walk(fear_dir):  
    	fear_audio = [MonoLoader(filename=fear_dir+'/'+fear_f)() for fear_f in sounds]
    fear_N = min([len(i) for i in fear_audio])  
    fear_samples = [i[:fear_N]/i.max() for i in fear_audio]  
    fear_x = np.array(fear_samples).sum(axis=0) 
    fear_X = 0.5*fear_x/fear_x.max()
    fear_Harmonic, fear_Percussive = decompose.hpss(librosa.core.stft(fear_X))
    fear_harmonic = istft(fear_Harmonic) 
    MonoWriter(filename=files_dir+'/mfcc/angry/angry_mix_mfcc.ogg', format = 'ogg', sampleRate = 44100)(fear_harmonic)
  

    happiness_dir = files_dir+'/mfcc/relaxed'
    for subdirs, dirs, happy_sounds in os.walk(happiness_dir):
    	happy_audio = [MonoLoader(filename=happiness_dir+'/'+happy_f)() for happy_f in happy_sounds]
    happy_N = min([len(i) for i in happy_audio])  
    happy_samples = [i[:happy_N]/i.max() for i in happy_audio]  
    happy_x = np.array(happy_samples).sum(axis=0) 
    happy_X = 0.5*happy_x/happy_x.max()
    happy_Harmonic, happy_Percussive = decompose.hpss(librosa.core.stft(happy_X))
    happy_harmonic = istft(happy_Harmonic)  
    MonoWriter(filename=files_dir+'/mfcc/relaxed/relaxed_mix_mfcc.ogg', format = 'ogg', sampleRate = 44100)(happy_harmonic) 
 

def centroid_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                          
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is angry")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/centroid/angry')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/centroid/relaxed')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.ogg', files_dir+'/centroid/angry/'+(str(e))+'mfcc.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.ogg', files_dir+'/centroid/relaxed/'+(str(e))+'mfcc.ogg')


    fear_dir = files_dir+'/centroid/angry' 
    for subdirs, dirs, sounds in os.walk(fear_dir):  
    	fear_audio = [MonoLoader(filename=fear_dir+'/'+fear_f)() for fear_f in sounds]
    fear_N = min([len(i) for i in fear_audio])  
    fear_samples = [i[:fear_N]/i.max() for i in fear_audio]  
    fear_x = np.array(fear_samples).sum(axis=0) 
    fear_X = 0.5*fear_x/fear_x.max()
    fear_Harmonic, fear_Percussive = decompose.hpss(librosa.core.stft(fear_X))
    fear_harmonic = istft(fear_Harmonic) 
    MonoWriter(filename=files_dir+'/centroid/angry/angry_mix_centroid.ogg', format = 'ogg', sampleRate = 44100)(fear_harmonic)
  

    happiness_dir = files_dir+'/centroid/relaxed'
    for subdirs, dirs, happy_sounds in os.walk(happiness_dir):
    	happy_audio = [MonoLoader(filename=happiness_dir+'/'+happy_f)() for happy_f in happy_sounds]
    happy_N = min([len(i) for i in happy_audio])  
    happy_samples = [i[:happy_N]/i.max() for i in happy_audio]  
    happy_x = np.array(happy_samples).sum(axis=0) 
    happy_X = 0.5*happy_x/happy_x.max()
    happy_Harmonic, happy_Percussive = decompose.hpss(librosa.core.stft(happy_X))
    happy_harmonic = istft(happy_Harmonic)  
    MonoWriter(filename=files_dir+'/centroid/relaxed/relaxed_mix_centroid.ogg', format = 'ogg', sampleRate = 44100)(happy_harmonic) 

def hfc_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is not sad")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is sad")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/hfc/not sad')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/hfc/sad')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/hfc/'+(str(e))+'hfc.ogg', files_dir+'/hfc/not sad/'+(str(e))+'hfc.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/hfc/'+(str(e))+'hfc.ogg', files_dir+'/hfc/sad/'+(str(e))+'hfc.ogg')


    sad_dir = files_dir+'/hfc/sad' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_N = min([len(i) for i in sad_audio])  
    sad_samples = [i[:sad_N]/i.max() for i in sad_audio]  
    sad_x = np.array(sad_samples).sum(axis=0) 
    sad_X = 0.5*sad_x/sad_x.max()
    sad_Harmonic, sad_Percussive = decompose.hpss(librosa.core.stft(sad_X))
    sad_percussive = istft(sad_Percussive) 
    MonoWriter(filename=files_dir+'/hfc/sad/sad_mix_hfc.ogg', format = 'ogg', sampleRate = 44100)(sad_percussive)
  

    not_sad_dir = files_dir+'/hfc/not sad'
    for subdirs, dirs, not_sad_sounds in os.walk(not_sad_dir):
    	not_sad_audio = [MonoLoader(filename=not_sad_dir+'/'+not_sad_f)() for not_sad_f in not_sad_sounds]
    not_sad_N = min([len(i) for i in not_sad_audio])  
    not_sad_samples = [i[:not_sad_N]/i.max() for i in not_sad_audio]  
    not_sad_x = np.array(not_sad_samples).sum(axis=0) 
    not_sad_X = 0.5*not_sad_x/not_sad_x.max()
    not_sad_Harmonic, not_sad_Percussive = decompose.hpss(librosa.core.stft(not_sad_X))
    not_sad_percussive = istft(not_sad_Percussive)  
    MonoWriter(filename=files_dir+'/hfc/not sad/not_sad_mix_hfc.ogg', format = 'ogg', sampleRate = 44100)(not_sad_percussive) 

def loudness_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is angry")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is not happy")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/loudness/angry')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/loudness/not happy')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/loudness/'+(str(e))+'loudness.ogg', files_dir+'/loudness/angry/'+(str(e))+'loudness.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/loudness/'+(str(e))+'loudness.ogg', files_dir+'/loudness/not happy/'+(str(e))+'loudness.ogg')


    sad_dir = files_dir+'/loudness/not happy' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_N = min([len(i) for i in sad_audio])  
    sad_samples = [i[:sad_N]/i.max() for i in sad_audio]  
    sad_x = np.array(sad_samples).sum(axis=0) 
    sad_X = 0.5*sad_x/sad_x.max()
    sad_Harmonic, sad_Percussive = decompose.hpss(librosa.core.stft(sad_X))
    sad_harmonic = istft(sad_Harmonic) 
    MonoWriter(filename=files_dir+'/loudness/not happy/not_happy_mix_loudness.ogg', format = 'ogg', sampleRate = 44100)(sad_harmonic)
  

    angry_dir = files_dir+'/loudness/angry'
    for subdirs, dirs, angry_sounds in os.walk(angry_dir):
    	angry_audio = [MonoLoader(filename=angry_dir+'/'+angry_f)() for angry_f in angry_sounds]
    angry_N = min([len(i) for i in angry_audio])  
    angry_samples = [i[:angry_N]/i.max() for i in angry_audio]  
    angry_x = np.array(angry_samples).sum(axis=0) 
    angry_X = 0.5*angry_x/angry_x.max()
    angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(angry_X))
    angry_harmonic = istft(angry_Harmonic)  
    MonoWriter(filename=files_dir+'/loudness/angry/angry_mix_loudness.ogg', format = 'ogg', sampleRate = 44100)(angry_harmonic) 

def inharmonicity_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is not relaxed")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is not angry")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/inharmonicity/not relaxed')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/inharmonicity/not angry')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/inharmonicity/'+(str(e))+'inharmonicity.ogg', files_dir+'/inharmonicity/not relaxed/'+(str(e))+'inharmonicity.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/inharmonicity/'+(str(e))+'inharmonicity.ogg', files_dir+'/inharmonicity/not angry/'+(str(e))+'inharmonicity.ogg')


    sad_dir = files_dir+'/inharmonicity/not relaxed' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_N = min([len(i) for i in sad_audio])  
    sad_samples = [i[:sad_N]/i.max() for i in sad_audio]  
    sad_x = np.array(sad_samples).sum(axis=0) 
    sad_X = 0.5*sad_x/sad_x.max()
    sad_Harmonic, sad_Percussive = decompose.hpss(librosa.core.stft(sad_X))
    sad_harmonic = istft(sad_Harmonic) 
    MonoWriter(filename=files_dir+'/inharmonicity/not relaxed/not_relaxed_mix_inharmonicity.ogg', format = 'ogg', sampleRate = 44100)(sad_harmonic)
  

    not_angry_dir = files_dir+'/inharmonicity/not angry'
    for subdirs, dirs, not_angry_sounds in os.walk(not_angry_dir):
    	not_angry_audio = [MonoLoader(filename=not_angry_dir+'/'+not_angry_f)() for not_angry_f in not_angry_sounds]
    not_angry_N = min([len(i) for i in not_angry_audio])  
    not_angry_samples = [i[:not_angry_N]/i.max() for i in not_angry_audio]  
    not_angry_x = np.array(not_angry_samples).sum(axis=0) 
    not_angry_X = 0.5*not_angry_x/not_angry_x.max()
    not_angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(not_angry_X))
    not_angry_harmonic = istft(not_angry_Harmonic)  
    MonoWriter(filename=files_dir+'/inharmonicity/not angry/not_angry_mix_inharmonicity.ogg', format = 'ogg', sampleRate = 44100)(not_angry_harmonic)

def contrast_emotions_remix(files_dir, negative_emotion_files, positive_emotion_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param negative_emotion_files: files with negative value                                    
    :param positive_emotion_files: files with positive value                                                                                   
    :param files_dir: data tag directory                                           
    """                                                                                         

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"emotion is not relaxed")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"emotion is not angry")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(positive_emotion_files)
    files_2 = set(sound_names).intersection(negative_emotion_files)
                                                           
    if files_1:                                               
        os.mkdir(files_dir+'/valleys/not relaxed')                                    
                                      
    if files_2:                                               
        os.mkdir(files_dir+'/valleys/not angry')                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/valleys/'+(str(e))+'contrast.ogg', files_dir+'/valleys/not relaxed/'+(str(e))+'contrast.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/valleys/'+(str(e))+'contrast.ogg', files_dir+'/valleys/not angry/'+(str(e))+'contrast.ogg')


    sad_dir = files_dir+'/valleys/not relaxed' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_N = min([len(i) for i in sad_audio])  
    sad_samples = [i[:sad_N]/i.max() for i in sad_audio]  
    sad_x = np.array(sad_samples).sum(axis=0) 
    sad_X = 0.5*sad_x/sad_x.max()
    sad_Harmonic, sad_Percussive = decompose.hpss(librosa.core.stft(sad_X))
    sad_harmonic = istft(sad_Harmonic) 
    MonoWriter(filename=files_dir+'/valleys/not relaxed/not_relaxed_mix_contrast.ogg', format = 'ogg', sampleRate = 44100)(sad_harmonic)
  

    not_angry_dir = files_dir+'/valleys/not angry'
    for subdirs, dirs, not_angry_sounds in os.walk(not_angry_dir):
    	not_angry_audio = [MonoLoader(filename=not_angry_dir+'/'+not_angry_f)() for not_angry_f in not_angry_sounds]
    not_angry_N = min([len(i) for i in not_angry_audio])  
    not_angry_samples = [i[:not_angry_N]/i.max() for i in not_angry_audio]  
    not_angry_x = np.array(not_angry_samples).sum(axis=0) 
    not_angry_X = 0.5*not_angry_x/not_angry_x.max()
    not_angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(not_angry_X))
    not_angry_harmonic = istft(not_angry_Harmonic)  
    MonoWriter(filename=files_dir+'/valleys/not angry/not_angry_mix_contrast.ogg', format = 'ogg', sampleRate = 44100)(not_angry_harmonic) 

#locate all files in data emotions dir
def multitag_emotions_dir(tags_dirs, negative_emotion_files, positive_emotion_files, neg_arous_dir, pos_arous_dir):
    """                                                                                     
    remix all files according to multitag emotions classes                                

    :param tags_dirs: directories of tags in data                                                                                            
    :param negative_emotion_files: files with multitag negative value
    :param positive_emotion_files: files with multitag positive value
    :param neg_arous_dir: directory where sounds with negative arousal value will be placed
    :param pos_arous_dir: directory where sounds with positive arousal value will be placed
                                                                                                                                                                         
    """                                                                                         
    files_format = ['.mp3', '.ogg', '.undefined', '.wav']

    if positive_emotion_files:
        print (repr(positive_emotion_files)+"By arousal, emotion is happy and angry, not sad and not relaxed")

    if negative_emotion_files:
        print (repr(negative_emotion_files)+"By arousal, emotion is sad and relaxed, not happy and not relaxed")

    sounds = []
                                                  
    for tag in tags_dirs:
            for types in next(os.walk(tag)):
               for t in types:
                   if os.path.splitext(t)[1] in files_format:
                       sounds.append(t)

    sound_names = []

    for s in sounds:
         sound_names.append(s.split('.')[0])                                                  
                 
    files_1 = set(sound_names).intersection(negative_emotion_files)
    files_2 = set(sound_names).intersection(positive_emotion_files) 

    if not os.path.exists(neg_arous_dir):
        os.makedirs(neg_arous_dir)
    if not os.path.exists(pos_arous_dir):
        os.makedirs(pos_arous_dir)                                                                     
                                        
    for tag in tags_dirs:
                 for types in next(os.walk(tag)):
                    for t in types:
                        if os.path.splitext(t)[0] in files_1:
                            if t in types:
                                shutil.copy(os.path.join(tag, t), os.path.join(neg_arous_dir,t))
                        if os.path.splitext(t)[0] in files_2:
                            if t in types:
                                shutil.copy(os.path.join(tag, t), os.path.join(pos_arous_dir,t)) 

from transitions import Machine
import random
import subprocess

neg_arous_dir = 'data/emotions/negative_arousal'   #directory where all data with positive arousal value will be placed                    
pos_arous_dir = 'data/emotions/positive_arousal'   #directory where all data with negative arousal value will be placed

#Johnny, Music Emotional State Machine
class MusicEmotionStateMachine(object):
            states = ['angry','sad','relaxed','happy','not angry','not sad', 'not relaxed','not happy']
            def __init__(self, name):
                self.name = name
                self.machine = Machine(model=self, states=MusicEmotionStateMachine.states, initial='angry')
            def sad_music_remix(self, neg_arous_dir, harmonic = None):
                    for subdirs, dirs, sounds in os.walk(neg_arous_dir):                            
                             negative_arousal_audio = [MonoLoader(filename=neg_arous_dir+'/'+s)() for s in sounds]                        
                    negative_arousal_N = min([len(i) for i in negative_arousal_audio])                                            
                    negative_arousal_samples = [i[:negative_arousal_N]/i.max() for i in negative_arousal_audio]  
                    negative_arousal_x = np.array(negative_arousal_samples).sum(axis=0)                     
                    negative_arousal_X = 0.5*negative_arousal_x/negative_arousal_x.max()
                    negative_arousal_Harmonic, negative_arousal_Percussive = decompose.hpss(librosa.core.stft(negative_arousal_X))
                    if harmonic is True:
                    	return negative_arousal_Harmonic
                    if harmonic is False or harmonic is None:
                    	sad_percussive = istft(negative_arousal_Percussive)
                    	MonoWriter(filename='data/emotions/sad/multitag_remix.ogg', format = 'ogg', sampleRate = 44100)(sad_percussive)
                    	subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/sad/multitag_remix.ogg'])
            def happy_music_remix(self, pos_arous_dir, harmonic = None):
                for subdirs, dirs, sounds in os.walk(pos_arous_dir):  
                    positive_arousal_audio = [MonoLoader(filename=pos_arous_dir+'/'+s)() for s in sounds]
                positive_arousal_N = min([len(i) for i in positive_arousal_audio])  
                positive_arousal_samples = [i[:positive_arousal_N]/i.max() for i in positive_arousal_audio]  
                positive_arousal_x = np.array(positive_arousal_samples).sum(axis=0) 
                positive_arousal_X = 0.5*positive_arousal_x/positive_arousal_x.max()
                positive_arousal_Harmonic, positive_arousal_Percussive = decompose.hpss(librosa.core.stft(positive_arousal_X))
		if harmonic is True:
			return positive_arousal_Harmonic
		if harmonic is False or harmonic is None:
		        happy_percussive = istft(positive_arousal_Percussive)
		        MonoWriter(filename='data/emotions/happy/multitag_remix.ogg', format = 'ogg', sampleRate = 44100)(happy_percussive)
		        subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/happy/multitag_remix.ogg'])
            def relaxed_music_remix(self, neg_arous_dir):
                neg_arousal_h = MusicEmotionStateMachine('remix').sad_music_remix(neg_arous_dir, harmonic = True)
                relaxed_harmonic = istft(neg_arousal_h)
                MonoWriter(filename='data/emotions/relaxed/multitag_remix.ogg', format = 'ogg', sampleRate = 44100)(relaxed_harmonic)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/relaxed/multitag_remix.ogg'])
            def angry_music_remix(self, pos_arous_dir):
                pos_arousal_h = MusicEmotionStateMachine('remix').happy_music_remix(pos_arous_dir, harmonic = True)
                angry_harmonic = istft(pos_arousal_h)
                MonoWriter(filename='data/emotions/angry/multitag_remix.ogg', format = 'ogg', sampleRate = 44100)(angry_harmonic)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/angry/multitag_remix.ogg'])
            def not_happy_music_remix(self, neg_arous_dir):
                for subdirs, dirs, sounds in os.walk(neg_arous_dir):  
                    	x = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:-1]))()
                    	y = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0] 
                if x_tempo < y_tempo:
                    y = effects.time_stretch(y, x_tempo/y_tempo)
                    if y.size < x.size:                                   
                        y = np.resize(y, x.size)
                    else:                                   
                        x = np.resize(x, y.size)
                else:
                    pass
                if x_tempo > y_tempo:
                    x = effects.time_stretch(x, y_tempo/x_tempo)
                    if x.size < y.size:                                           
                        x = np.resize(x, y.size)
                    else:                                   
                        y = np.resize(y, x.size)
                not_happy_x = np.sum((x,y),axis=0) 
                not_happy_X = 0.5*not_happy_x/not_happy_x.max()
                MonoWriter(filename='data/emotions/not happy/multitag_remix.ogg', format = 'ogg', sampleRate = 44100)(not_happy_X)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/not happy/multitag_remix.ogg'])
            def not_sad_music_remix(self, pos_arous_dir):
                for subdirs, dirs, sounds in os.walk(pos_arous_dir):  
                    	x = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:-1]))()
                    	y = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0] 
                if x_tempo < y_tempo:
                    x = effects.time_stretch(x, y_tempo/x_tempo)
                    if y.size < x.size:                                   
                        y = np.resize(y, x.size)
                    else:                                   
                        x = np.resize(x, y.size)
                else:
                    pass
                if x_tempo > y_tempo:
                    y = effects.time_stretch(y, x_tempo/y_tempo)
                    if x.size < y.size:                                           
                        x = np.resize(x, y.size)
                    else:                                   
                        y = np.resize(y, x.size)
                not_sad_x = np.sum((x,y),axis=0) 
                not_sad_X = 0.5*not_sad_x/not_sad_x.max()
                MonoWriter(filename='data/emotions/not sad/multitag_remix.ogg', format = 'ogg', sampleRate = 44100)(not_sad_X)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/not sad/multitag_remix.ogg'])
            def not_angry_music_remix(self, neg_arous_dir):
                for subdirs, dirs, sounds in os.walk(neg_arous_dir):  
                    	x = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:-1]))()
                    	y = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0] 
                morph = stft.morph(x1 = x,x2 = y,fs = 44100,w1=np.hanning(1025),N1=2048,w2=np.hanning(1025),N2=2048,H1=512,smoothf=0.1,balancef=0.7)
                write_wav(morph,44100, 'data/emotions/not angry/multitag_remix.wav') 
                subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/not angry/multitag_remix.wav'])
            def not_relaxed_music_remix(self, pos_arous_dir):
                for subdirs, dirs, sounds in os.walk(pos_arous_dir):  
                    	x = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:-1]))()
                    	y = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0] 
                morph = stft.morph(x1 = x,x2 = y,fs = 44100,w1=np.hanning(1025),N1=2048,w2=np.hanning(1025),N2=2048,H1=512,smoothf=0.01,balancef=0.7)
                write_wav(morph,44100, 'data/emotions/not relaxed/multitag_remix.wav') 
                subprocess.call(["ffplay", "-nodisp", "-autoexit", 'data/emotions/not relaxed/multitag_remix.wav'])


Usage = "./MusicEmotionMachine.py [FILES_DIR] [MULTITAG CLASSIFICATION False/True/None]"
if __name__ == '__main__':
  
    if len(sys.argv) < 3:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        files_dir = sys.argv[1]

        if not os.path.exists(files_dir):                         
            raise IOError("Must run MIR analysis") 

        if sys.argv[2] in ('True'):
            neg_and_pos = multitag_emotion_classifier(tags_dirs)
            emotions_data_dir()
            multitag_emotions_dir(tags_dirs, neg_and_pos[0], neg_and_pos[1], neg_arous_dir, pos_arous_dir)
        if sys.argv[2] in ('None'):
            neg_and_pos = plot_emotion_clusters(files_dir, multitag = None)
            bpm_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            attack_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            dissonance_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            mfcc_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            centroid_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            hfc_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            loudness_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            inharmonicity_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
            contrast_emotions_remix(files_dir, neg_and_pos[0], neg_and_pos[1])
        if sys.argv[2] in ('False'): 
		me = MusicEmotionStateMachine("Johnny") #calling Johnny                        
		me.machine.add_ordered_transitions()    #Johnny is very sensitive                  
		                                                          
		while(1):                                                 
		    me.next_state()                                       
		    if me.state == random.choice(me.states):              
		        if me.state == 'happy':                           
		            print me.state                                
		            me.happy_music_remix(pos_arous_dir)
		        if me.state == 'sad':                  
		            print me.state                     
		            me.sad_music_remix(neg_arous_dir)  
		        if me.state == 'angry':                
		            print me.state                     
		            me.angry_music_remix(pos_arous_dir)
		        if me.state == 'relaxed':              
		            print me.state                     
		            me.relaxed_music_remix(neg_arous_dir)
		        if me.state == 'not happy':              
		            print me.state                     
		            me.not_happy_music_remix(neg_arous_dir)
		        if me.state == 'not sad':              
		            print me.state                     
		            me.not_sad_music_remix(pos_arous_dir)
		        if me.state == 'not angry':              
		            print me.state                     
		            me.not_angry_music_remix(neg_arous_dir)
		        if me.state == 'not relaxed':              
		            print me.state                     
		            me.not_angry_music_remix(pos_arous_dir)
                                                       
    except Exception, e:                     
        logger.exception(e)
