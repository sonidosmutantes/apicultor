#! /usr/bin/env python
# -*- coding: utf-8 -*-

from SoundSimilarity import get_files, get_dics, desc_pair
import time
from colorama import Fore
from collections import Counter
from itertools import combinations
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                           
from essentia.standard import *
from smst.utils.audio import write_wav    
from sklearn import svm, preprocessing
from sklearn.decomposition.pca import PCA
from sklearn.cluster import KMeans
from sklearn.utils.extmath import safe_sparse_dot as ssd
import librosa
from librosa import *                                       
import shutil
from smst.models import stft 
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)                               

#emotion classification
class descriptors_and_keys():
    """
    get all descriptions given descriptor keys
    :param files_dir: data tag dir if not performing multitag classification. data dir if performing multitag classification
    :param multitag: if True, will classify all downloaded files and remix when performing emotional state transition 
    """                    
    def __init__(self, files_dir,  multitag):
        self.multitag = None
        self._files_dir = files_dir
        if multitag == None or multitag == False:                                                
            self._files = get_files(files_dir)                                    
            self._dics = get_dics(files_dir)                                      
        elif multitag == True:                                              
            self._files = np.hstack([get_files(tag) for tag in files_dir])        
            self._dics = np.hstack([get_dics(tag) for tag in files_dir])          
        for i,x in enumerate(np.vstack((self._files, [len(i) for i in self._dics])).T): 
            if x[1] != str(13): #if dictionary of a file is shorter...     
                fname = i                                                  
                for i in self._files:                                            
                    fname = i[fname] #...finds the name of the .json file  
                for tag in tags_dirs:                                      
                    for subdir, dirs, sounds in os.walk(tag+'/descriptores'):
                        for s in sounds:                                   
                            if fname == s:                                 
                                print (os.path.abspath(tag+'/descriptores/'+s) +" has less descriptors and will be discarded for clustering and classification. This happens because there was an error in MIR analysis, so the sound has to be analyzed again after being processed") 
                                os.remove((os.path.abspath(tag+'/descriptores/'+s))) #... and deletes the less suitable .json file
        self._files = list(self._files[0])
        self._dics = list(self._dics)
        duplicate = [] 
        self._duplicate_index = []
        for i in range(len(self._files)):                             
            if self._files[i].endswith('(1).json') is True:
                self._duplicate_index.append(i)
        for i in self._files:                             
            if i.endswith('(1).json') is True:
                duplicate.append(i)
        for i in duplicate:                         
            for j in self._files:                   
                if j == i:         
                    self._files.remove(j)
        for i in self._duplicate_index:
                self._dics.remove(self._dics[i])
        self._files_features = desc_pair(self._files,self._dics).files_features
        self._keys = [3,9,0,2,1,7,4,10,5,8,11,12]  
        self._features = []
        for i in range(len(self._keys)):
            self._features.append(np.float64(zip(*self._files_features)[self._keys[i]]))    
                                                   
class features_combinations():
    """
    combine features randomly to get most possible combinations
    :param features: features used for classification task                                                                  
    """  
    def __init__(self, features):
        self._features = features
        self._features_combination = []
        for i in range(len(self._features)):
            self._features_combination.append(random.choice(list(combinations(features,6))))
        self._random_pairs = []
        for i in range(0, len(self._features_combination)):
            for j in range(len(self._features_combination[i])):
                self._random_pairs.append(np.vstack((self._features_combination[i][j], self._features_combination[i][j])).T)     

def feature_scaling(features):
    """
    classify sounds based on emotivity (emotive or non-emotive) using Support Vector Machines
    :param features: combinations of features                                                                               
    :returns:                                                                                                         
      - fscaled: scaled features
    """      
    fscaled = []
    for i in range(len(features)):    
        fscaled.append(PCA(n_components = 2, whiten = True).fit_transform(features[i]-np.mean(features[i])/(np.max(features[i])-np.min(features[i]))))                            
    return fscaled 

def KMeans_clusters(fscaled):
    """
    KMeans clustering for features                                                           
    :param fscaled: scaled features                                                                                         
    :returns:                                                                                                         
      - labels: classes         
    """
    labels = []
    for i in range(len(fscaled)):
        labels.append(KMeans(init = PCA(n_components = 2, whiten = True).fit(fscaled[i]).components_, n_clusters=2, n_init=1, precompute_distances = True).fit(fscaled[i]).labels_)
    return labels

class deep_support_vector_machines(object):
    """
    Functions for Deep Support Vector Machines                                               
    :param features: scaled features                                                                                        
    :param labels: classes                                                                                            
    """ 
    def __init__(self, features, labels):
        return self
    def polynomial_kernel(self, x, y):
	"""
	Custom polynomial kernel function, similarities of vectors over polynomials
	:param x: array of input vectors
	:param y: array of input vectors
	:returns:
	  - pk: inner product
	"""
        c = 0.5

        degree = 5

        gamma = 1.0/x.shape[1]

        pk = ssd(x, y.T, dense_output = True)

        pk *= gamma

        while np.all(pk + c > 1) == True:
            c -= 0.1
            if c == 0:
                pass

        pk += c

        while np.all(pk ** degree == 0.0) == True:
           degree -= 1
           if degree == 2:
                pass

        pk **= degree
        return pk
    def support_vectors(self, layer_support, data):
	"""
	Computes vector outputs and labels
	:param layer_support: support vectors (by index)
	:param data: input data (features)
	  - support_vectors: the values of the support vectors
	"""
        support_vectors = []
        for i in layer_support:
            support_vectors.append(data[i])
        return support_vectors 
    def gradient_ascent(self, features,labels, w): 
	"""
	Finds local maximum for vectors
	:param features: features
	:param labels: classes
	:param w: theta
	:returns:
	  - theta: w
	"""              
        theta = w   
        for j in range(0, 150):
            dataIndex = range(features.shape[0])
            for i in range(features.shape[0]):
                alpha = 4/(1.0+j+i)+0.01
                randIndex = int(random.uniform(0,len(dataIndex)))
                h = 1 / (1 + np.exp(1-(sum(features[randIndex]*theta))))  
                error = labels[randIndex] - h
                theta = theta + (alpha * features[randIndex] * error) - (2 * np.mean(theta))
                del(dataIndex[randIndex])
                if np.all(theta >= 0.1) == True:                                   
                    return theta      
                if np.all(theta >= 0.1) == False:      
                    continue   
        return theta
    def gradient_descent(self, features, labels, w):
	"""
	Finds local minimum for vectors
	:param features: features
	:param labels: classes
	:param w: theta
	:returns:
	  - theta: w
	"""    
        theta = self.gradient_ascent(features,labels, w)
        transposef = features.T    
        for i in range(0,20000):
            h = np.dot(features, theta)
            loss = h - labels
            gradients = []
            n = float(len(labels))
            for j in range(len(theta)):
                gradients.append(np.sum(loss * features[:,j]) / n)
            cost = np.sum(loss**2)
            gradient = np.dot(transposef, loss)/features.shape[0]
            theta = [theta - 0.02 * g for theta, g in zip(theta, gradients)]
            new_h = np.dot(features, theta)
            new_loss = new_h - labels             
            new_cost = np.sum(new_loss**2)
            if i>=5 and (new_cost - cost <= 1e-12 and  new_cost - cost >= -1e12):
                return theta
        return theta 
    def sgd_approximation(self, S, labl): 
	"""
	Stochastic approximations given features and labels (useful when best labels don't necessarilly match)
	:param S: features
	:param labl: classes
	:returns:
	  - S: features
	  - lab: labels
	"""                   
        from sklearn.linear_model import SGDClassifier as sgd                  
        lab = sgd().fit(S,labl)                             
        w = lab.coef_                       
        lab = lab.predict(S*w)         
        S = S*w                
        return S, lab
 
class svm_layers(deep_support_vector_machines):
    """
    Functions for Deep Support Vector Machines layers                                        
    :param features: scaled features                                                                                        
    :param labels: classes                                                                                            
    """    
    def __init__(self):
        super(deep_support_vector_machines, self).__init__()
    def layer_computation(self, features, labels):
	"""
	Computes vector outputs and labels
	:param features: scaled features
	:param labels: classes
	:returns:
	  - S: arrays of outputs of the layers
	  - labels_to_file: classes from all layers
	  - scores: prediction scores
	"""
        S = []
        scores = []
        target_labels = []
        for i,j in enumerate(features):
            layer = svm.SVC(kernel = self.polynomial_kernel, shrinking = False, tol = 1e-12, decision_function_shape = 'ovr', cache_size = 512, C = 1.0/(2*features[i].shape[0]), verbose = True)
            layer.fit(features[i], labels[i])
            sv = self.support_vectors(layer.support_, features[i])
            w = layer.dual_coef_.dot(sv)[0]
            layer_input_labels = layer.predict(features[i])
            theta = self.gradient_descent(features[i], layer_input_labels, w)
            s = ( (theta * features[i]) - (theta / features[i]) ) 
            layer_labels = layer.predict(s)
            target_labels.append(layer_labels)
            scores.append(layer.score(s, layer_labels))
            S.append( s )
        labels_to_file = np.array(target_labels).T
        return S, labels_to_file, scores                  
    def sum_of_S(self, S):
	"""
	Sums the vector outputs of Deep Support Vector Machine layers
	:param S: arrays of vector outputs
	:returns:
	  - sum of outputs for the main layer
	"""            
        return np.sum(S,axis=0)
    def best_labels(self, labels_to_file):
	"""
	Get the best labels for regression
	:param labels_to_file: the classes given to the outputs of the layers
	:returns:
	  - labl: the best labels to apply regression to the sum of outputs of the layers
	"""
        count_labels = [Counter(list(i)).most_common() for i in labels_to_file]
        labl = []
        for i in count_labels:
            labl.append(i[0][0])
        labl = np.array(labl).T
        return labl

class main_svm(deep_support_vector_machines):
    """
    Functions for the main layer of the Deep Support Vector Machine                          
    :param S: sum of layers outputs                                                                                         
    :param labels: best classes from outputs                                                                          
    :returns:                   
      - negative_emotion_files: files with negative emotional value (emotional meaning according to the whole performance)
      - positive_emotion_files: files with positive emotional value (emotional meaning according to the whole performance) 
    """
    def __init__(self, S, lab):
        super(deep_support_vector_machines, self).__init__()
        self._main_svm = svm.SVC(kernel = 'linear', degree = 8, decision_function_shape = 'ovr', coef0 = 1, shrinking = True, class_weight = 'balanced', tol = 1e-12, cache_size = 512, C=1.0/(2*S.shape[0]), verbose = True).fit(S, lab)
        self._labels = self._main_svm.predict(S)
        self._w = self._main_svm.coef_[0]                      
        self._a = -self._w[0] / self._w[1]                                                    
        self._xx = np.linspace(S[:,0].min(), S[:,1].max())                                                                          
        self._yy = self._a * self._xx + (self._main_svm.intercept_[0])/self._w[1]# this intercept finds it out correctly in polynomial
        #calculate the parallels of separation                                                                                               
        self._b = self._main_svm.support_vectors_[0]                                                                                          
        self._yy_down = self._a * self._xx + (self._b[1] - self._a * self._b[0])                                                              
        self._b = self._main_svm.support_vectors_[-1]                                                                                         
        self._yy_up = self._a * self._xx + (self._b[1] - self._a * self._b[0])                                 
    def plot_emotion_classification(self):
	"""
	3D plotting of the classfication
	"""
            #3D plotting                                 
        from mpl_toolkits.mplot3d import Axes3D 
               
        fig = plt.figure()                                                                                                                    
        ax = Axes3D(fig)                                                                         
        ax.plot3D(self._xx, self._yy, 'k-')                                                                                                     
        ax.plot3D(self._xx, self._yy_down, 'k--')                                                                                               
        ax.plot3D(self._xx, self._yy_up, 'k--')                                                                                                 
        ax.scatter3D(S[:, 0], S[:, 1], c=self._labels, cmap=plt.cm.Paired) 
       
        print (Fore.WHITE + "El grupo negativo '0' esta coloreado en azul, el grupo positivo '1' esta coloreado en rojo") 
                    
        time.sleep(2)                                                  
        plt.show() 
                                                    
    def neg_and_pos(self, files):
	"""
	Lists of files according to emotion label (0 for negative, 1 for positive)
	:param files: filenames of inputs
	:returns:
	  - negative_emotion_files: files with negative emotional value (emotional meaning according to the whole performance)
	  - positive_emotion_files: files with positive emotional value (emotional meaning according to the whole performance) 
	"""
        negative_emotion_group = map(lambda json: files[json], [i for i, x in enumerate(self._labels) if x ==0])
        negative_emotion_files = [i.split('.json')[0] for i in negative_emotion_group]
        positive_emotion_group = map(lambda json: files[json], [i for i, x in enumerate(self._labels) if x ==1]) 
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
tags_dirs = lambda files_dir: [os.path.join(files_dir,dirs) for dirs in next(os.walk(os.path.abspath(files_dir)))[1]]

#classify all downloaded audio in tags
def multitag_emotion_classifier(tags_dirs):
    """                                                                                     
    emotion classification of all data                                
                                                                                            
    :param tags_dirs = paths of tags in data                                                                              
    """           
    neg_and_pos = deep_support_vector_machine(tags_dirs, multitag = True)   
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
    files_format = ['.mp3', '.ogg', '.undefined', '.wav', '.mid', '.wma', '.amr']

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

neg_arous_dir = 'data/emotions/negative_arousal'   #directory where all data with negative arousal value will be placed                    
pos_arous_dir = 'data/emotions/positive_arousal'   #directory where all data with positive arousal value will be placed

#Johnny, Music Emotional State Machine
class MusicEmotionStateMachine(object):
            states = ['angry','sad','relaxed','happy','not angry','not sad', 'not relaxed','not happy']
            def __init__(self, name):
                self.name = name
                self.machine = Machine(model=self, states=MusicEmotionStateMachine.states, initial='angry')
            def sad_music_remix(self, neg_arous_dir, harmonic = None):
                    for subdirs, dirs, sounds in os.walk(neg_arous_dir):                            
                             x = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:-1]))() 
                             y = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:]))() 
                    negative_arousal_audio = np.array((x,y))                        
                    negative_arousal_N = min([len(i) for i in negative_arousal_audio])                                            
                    negative_arousal_samples = [i[:negative_arousal_N]/i.max() for i in negative_arousal_audio]  
                    negative_arousal_x = np.array(negative_arousal_samples).sum(axis=0)                     
                    negative_arousal_X = 0.5*negative_arousal_x/negative_arousal_x.max()
                    negative_arousal_Harmonic, negative_arousal_Percussive = decompose.hpss(librosa.core.stft(negative_arousal_X))
                    if harmonic is True:
                    	return negative_arousal_Harmonic
                    if harmonic is False or harmonic is None:
                    	sad_percussive = istft(negative_arousal_Percussive)
                    	remix_filename = 'data/emotions/sad/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                    	MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(sad_percussive)
                    	subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def happy_music_remix(self, pos_arous_dir, harmonic = None):
                for subdirs, dirs, sounds in os.walk(pos_arous_dir):  
                    x = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:-1]))()
                    y = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:]))()
                positive_arousal_audio = np.array((x,y))
                positive_arousal_N = min([len(i) for i in positive_arousal_audio])  
                positive_arousal_samples = [i[:positive_arousal_N]/i.max() for i in positive_arousal_audio]  
                positive_arousal_x = np.array(positive_arousal_samples).sum(axis=0) 
                positive_arousal_X = 0.5*positive_arousal_x/positive_arousal_x.max()
                positive_arousal_Harmonic, positive_arousal_Percussive = decompose.hpss(librosa.core.stft(positive_arousal_X))
		if harmonic is True:
			return positive_arousal_Harmonic
		if harmonic is False or harmonic is None:
		        happy_percussive = istft(positive_arousal_Percussive)
		        remix_filename = 'data/emotions/happy/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
		        MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(happy_percussive)
		        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def relaxed_music_remix(self, neg_arous_dir):
                neg_arousal_h = MusicEmotionStateMachine('remix').sad_music_remix(neg_arous_dir, harmonic = True)
                relaxed_harmonic = istft(neg_arousal_h)
                remix_filename = 'data/emotions/relaxed/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(relaxed_harmonic)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def angry_music_remix(self, pos_arous_dir):
                pos_arousal_h = MusicEmotionStateMachine('remix').happy_music_remix(pos_arous_dir, harmonic = True)
                angry_harmonic = istft(pos_arousal_h)
                remix_filename = 'data/emotions/angry/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(angry_harmonic)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
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
                remix_filename = 'data/emotions/not happy/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, sampleRate = 44100)(not_happy_X)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
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
                remix_filename = 'data/emotions/not sad/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, sampleRate = 44100)(not_sad_X)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def not_angry_music_remix(self, neg_arous_dir):
                for subdirs, dirs, sounds in os.walk(neg_arous_dir):  
                    	x = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:-1]))()
                    	y = MonoLoader(filename=neg_arous_dir+'/'+random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0] 
                morph = stft.morph(x1 = x,x2 = y,fs = 44100,w1=np.hanning(1025),N1=2048,w2=np.hanning(1025),N2=2048,H1=512,smoothf=0.1,balancef=0.7)
                remix_filename = 'data/emotions/not angry/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                write_wav(morph,44100, remix_filename) 
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def not_relaxed_music_remix(self, pos_arous_dir):
                for subdirs, dirs, sounds in os.walk(pos_arous_dir):  
                    	x = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:-1]))()
                    	y = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0] 
                morph = stft.morph(x1 = x,x2 = y,fs = 44100,w1=np.hanning(1025),N1=2048,w2=np.hanning(1025),N2=2048,H1=512,smoothf=0.01,balancef=0.7)
                remix_filename = 'data/emotions/not relaxed/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                write_wav(morph,44100, remix_filename) 
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])


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
            tags_dirs = tags_dirs(files_dir)
            files = descriptors_and_keys(tags_dirs, True)._files
            features = descriptors_and_keys(tags_dirs, True)._features
            fscaled = feature_scaling(features_combinations(features)._random_pairs)
            labels = KMeans_clusters(fscaled)
            layers_outputs, labels_to_file, scores = svm_layers().layer_computation(fscaled, labels)
            S = svm_layers().sum_of_S(layers_outputs)
            labl = svm_layers().best_labels(labels_to_file)
            S, lab = svm_layers().sgd_approximation(S, labl)
            main_svm(S,lab).plot_emotion_classification()
            neg_and_pos = main_svm(S,lab).neg_and_pos(files)
            emotions_data_dir()
            multitag_emotions_dir(tags_dirs, neg_and_pos[0], neg_and_pos[1], neg_arous_dir, pos_arous_dir)
        if sys.argv[2] in ('None'):
            files = descriptors_and_keys(files_dir, None)._files
            features = descriptors_and_keys(files_dir, None)._features
            fscaled = feature_scaling(features_combinations(features)._random_pairs)
            labels = KMeans_clusters(fscaled)
            layers_outputs, labels_to_file, scores = svm_layers().layer_computation(fscaled, labels)
            S = svm_layers().sum_of_S(layers_outputs)
            labl = svm_layers().best_labels(labels_to_file)
            S, lab = svm_layers().sgd_approximation(S, labl)
            main_svm(S,lab).plot_emotion_classification()
            neg_and_pos = main_svm(S,lab).neg_and_pos(files)
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
		            me.not_relaxed_music_remix(pos_arous_dir)
                                                       
    except Exception, e:                     
        logger.exception(e)
