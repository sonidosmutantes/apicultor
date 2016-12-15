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
from python_toolbox import caching
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
        for i in reversed(self._duplicate_index):
                self._dics.remove(self._dics[i])
        funique = []
        indexes = []
        for i,x in enumerate(self._files):
            if x in funique:
                indexes.append(i)
            if x not in funique:
                funique.append(x)
        for i in reversed(indexes):
            self._dics.remove(self._dics[i])
        self._files = funique
        self._files_features = desc_pair(self._files,self._dics).files_features
        self._keys = [3,9,0,2,1,7,4,10,5,8,11,12]  
        self._features = []
        for i in range(len(self._keys)):
            self._features.append(np.float64(zip(*self._files_features)[self._keys[i]]))    
        self._features = np.array(self._features).T

def feature_scaling(f):
    """
    scale features
    :param features: combinations of features                                                                               
    :returns:                                                                                                         
      - fscaled: scaled features
    """    
    from sklearn.preprocessing import StandardScaler as stdscale 
    fscaled = stdscale().fit(f).transform(f)
    return fscaled

def KMeans_clusters(fscaled):
    """
    KMeans clustering for features                                                           
    :param fscaled: scaled features                                                                                         
    :returns:                                                                                                         
      - labels: classes         
    """
    labels = (KMeans(init = PCA(n_components = 2).fit(fscaled).components_, n_clusters=2, n_init=1, precompute_distances = True).fit(fscaled).labels_)
    return labels

class deep_support_vector_machines(object):
    """
    Functions for Deep Support Vector Machines                                               
    :param features: scaled features                                                                                        
    :param labels: classes                                                                                            
    """ 
    def __init__(self, kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    @classmethod
    @caching.cache(max_size=100)
    def polynomial_kernel(self, x, y, gamma):
	"""
	Custom polynomial kernel function, similarities of vectors over polynomials
	:param x: array of input vectors
	:param y: array of input vectors
	:param gamma: gamma
	:returns:
	  - pk: inner product
	"""
        c = 1
        degree = 2

        pk = ssd(x, y.T, dense_output = True)

        pk *= gamma

        pk += c

        pk **= degree
        return pk
    @classmethod
    @caching.cache(max_size=100)
    def linear_kernel_matrix(self, x, y):   
        return np.dot(x,y)
    @classmethod
    @caching.cache(max_size=100)
    def sigmoid_kernel(self, x, y, gamma):
	"""
	Custom sigmoid kernel function, similarities of vectors over polynomials
	:param x: array of input vectors
	:param y: array of input vectors
	:param gamma: gamma
	:returns:
	  - sk: inner product
	"""
        c = 1
        degree = 2

        sk = ssd(x, y.T, dense_output = True)

        sk *= gamma

        sk += c

        np.tanh(np.array(sk, dtype='float64'), np.array(sk, dtype = 'float64'))
        return np.array(sk, dtype = 'float64')
    @classmethod
    @caching.cache(max_size=100)
    def rbf_kernel(self, x, y, gamma):
	"""
	Custom sigmoid kernel function, similarities of vectors over polynomials
	:param x: array of input vectors
	:param y: array of input vectors
	:param sigma: array of input vectors
	:returns:
	  - rbfk: inner product
	"""    
        from sklearn.metrics.pairwise import euclidean_distances  
        rbfk = euclidean_distances(x, y, squared=True)  
        rbfk *= -gamma 
        np.exp(rbfk, rbfk)  
        return rbfk
    def fit_model(self, features,labels, kernel1, kernel2, C, reg_param, gamma):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.gamma = gamma

        x = np.ones(shape = (len(features),3)) 
        x[:,(0,1)] = features[:,(0,1)]  

        if np.all(labels[labels  > 0]) == False:                                      
            raise Exception("There has to be two or more targets")

        lab = labels * 2 - 1

        if self.kernel2 == "linear":                                      
            kernel = self.linear_kernel_matrix
            matrix = self.linear_kernel_matrix(x, x.T)
        if self.kernel2 == "poly":
            kernel = self.polynomial_kernel
            matrix = self.polynomial_kernel(x, x, gamma)
        if self.kernel2 == "rbf":                                      
            kernel = self.rbf_kernel
            matrix = self.rbf_kernel(x, x, gamma)
        if self.kernel2 == "sigmoid":
            kernel = self.sigmoid_kernel
            matrix = self.sigmoid_kernel(x, x, gamma)
        if self.kernel2 == None:              
            print ("Apply a kernel")

        Q = np.zeros((len(labels), len(labels)))
        for i in xrange(len(labels)):
            for j in xrange(i, len(labels)):
                Qvalue = lab[i] * lab[j]
                if kernel == self.linear_kernel_matrix:        
                    Qvalue *= kernel(features[i,:], features[j,:])     
                else:                    
                    Qvalue *= kernel(features[i,:], features[j,:], gamma)
                Q[i,j] = Q[j,i] = Qvalue

        class_weight = float(len(features))/(2 * np.bincount(labels))

        sample_weight = class_weight[labels]

        self.a = np.zeros(features.shape[0])
        max_iterations = len(features)
        iterations = 0
        diff = 1. + reg_param
        while diff > reg_param:
            data_i = range(len(features))
            for k in range(len(features)):
                a0 = 4./(1. + iterations + k) + self.a.copy()
                random_i = int(random.uniform(0,len(data_i)))
                self.a[random_i] = self.a[random_i] + (1/np.diag(matrix))[random_i] * (1-x[random_i,2]*sum(self.a*x[:,2]*matrix[:,random_i]))
                if self.a[random_i] < 0:                                       
                    self.a[random_i] = 0
                self.a[random_i] = min(self.a[random_i], C * sample_weight[i] * class_weight[labels[random_i]])
                diff = sum((self.a-a0)*(self.a-a0))
                del data_i[random_i]
                iterations += 1
                if diff < reg_param:                                       
                    break

        print ("Iterated " + str(iterations) +  " times for gradient ascent")

        self.w = np.zeros(features.shape[1])
        iterations = 0  
        delta = 1. + reg_param                   
        while delta > reg_param:
            data_i = range(len(features))
            for i in range(len(features)):
                delta = 4./(1. + iterations + i) + 0. 
                random_i = int(random.uniform(0,len(data_i)))             
                gradient = np.dot(Q[random_i,:], self.a) - 1.0                                                                         
                adelta = self.a[random_i] - min(max(self.a[random_i] - gradient/Q[random_i,random_i], 0.0), C * sample_weight[i] * class_weight[labels[random_i]])
                self.w += adelta * features[random_i,:]  
                delta += abs(adelta)
                self.a[random_i] -= adelta
                del data_i[random_i]
                iterations += 1
                if delta < reg_param:                                       
                    break

        print ("Iterated " + str(iterations) +  " times for gradient descent")

        if np.sum(self.w) == 0:
            raise Exception("Invalid value for weight, it should be higher than zero") 


        if np.sum(self.w) == np.nan:
            raise Exception("Invalid value for weight, it should be higher than zero and not nan") 


        print ("Weight coefficient = "), self.w

        self.svs = features[self.a > 0.0, :]

        self.ns = np.sort(list(set(np.where(self.a > 0.0)[0])))

        self.a = (self.a * lab)[self.a > 0.0]

        self.nvs = [0,0]
        for i in np.sign(self.a):                            
            if i == -1:
                self.nvs[0] += 1
            if i == 1:
                self.nvs[1] += 1

        print ("Number of Support Vectors is " + str(self.nvs[0]) + " for negative classes and " + str(self.nvs[1]) + " for positive classes")

        self.bias = self.partial_predictions(self.svs[0,:])[0]
        if self.a[0] > 0:                           
            self.bias *= -1

        print ("Bias value = "), self.bias

        return self.w, self.a, self.bias 
            
    def partial_predictions(self, features):
        if (len(features.shape)  < 2):
            features = features.reshape((1,-1))
        classes = np.zeros(len(features))
        for i in xrange(len(features)):
            for j in xrange(len(self.svs)):
                if self.kernel1 == "sigmoid":                           
                    classes[i] += self.a[j] * self.sigmoid_kernel(self.svs[j,:],features[i,:], self.gamma)                                      
                if self.kernel1 == "poly":                              
                    classes[i] += self.a[j] * self.polynomial_kernel(self.svs[j,:],features[i,:], self.gamma) 
                if self.kernel1 == "rbf":   
                    classes[i] += self.a[j] * self.rbf_kernel(self.svs[j,:],features[i,:], self.gamma)                                           
                if self.kernel1 == "linear":   
                    k_fun = self.kernel1     
                    classes[i] += self.a[j] * self.linear_kernel_matrix(self.svs[j,:],features[i,:])
        return classes

    def decision_function(self, features):
        if self.kernel1 == "linear":
            k = self.linear_kernel_matrix(self.svs, features.T)
        if self.kernel1 == "poly":
            k = self.polynomial_kernel(self.svs, features, self.gamma)
        if self.kernel1 == "sigmoid":
            k = self.sigmoid_kernel(self.svs, features, self.gamma)
        if self.kernel1 == "rbf":
            k = self.rbf_kernel(self.svs, features, self.gamma)
        start = [sum(self.nvs[:i]) for i in range(len(self.nvs))]
        end = [start[i] + self.nvs[i] for i in range(len(self.nvs))]
        a = np.array([list(self.a)])
        c = [ sum(a[ i ][p] * k[p] for p in range(start[j], end[j])) +
              sum(a[j-1][p] * k[p] for p in range(start[i], end[i]))
                for i in range(len(self.nvs)) for j in range(i+1,len(self.nvs))]

        return [sum(x) for x in zip(c, [self.bias])]

    def predictions(self, features):                   
        classes = self.decision_function(features)[0]  > 0 
        lab = []                   
        for c in classes:
            if c == True:
                lab.append(1)
            else:
                lab.append(0) 
        return lab

#classify all the features using different kernels (different products) 
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
	  - labels_to_file: potential classes
	"""
        #classes are predicted even if we only use the decision functions as part of the output data to get a better scope of classifiers
        from sklearn.metrics import accuracy_score
        fxs = []
        sample_weight = float(len(features)) / (len(np.array(list(set(labels)))) * np.bincount(labels))
        for i in range(0,4):
            print ("Calculating values for prediction")
            if i == 0:
                classes = labels
                self.fit_model(features[labels==0], labels[labels==0], "rbf", "sigmoid", 100, 0.1, 0.01)
                print ("Predicting")
                fx = self.decision_function(features)[0]
                classes[labels==0] = self.predictions(features[labels==0])
            if i == 1:
                classes = labels
                self.fit_model(features[labels==1], labels[labels==1], "linear", "poly", 20, 0.1, 0.035111917342151272)
                print ("Predicting")
                fx = self.decision_function(features)[0]
                classes[labels==1] = self.predictions(features[labels==1])
            if i == 2:
                self.fit_model(features, labels, "poly", "linear", 20, 0.10000000000000001, 0.035111917342151272)
                print ("Predicting")
                fx = self.decision_function(features)[0]
                classes = self.predictions(features)
            if i == 3:
                classes = labels
                self.fit_model(features, labels, "rbf", "poly", 100.0, 0.01, 9.9999999999999995e-07)
                print ("Predicting")
                fx = self.decision_function(features)[0]
                classes = self.predictions(features)
            print ("Predicted"), classes
            lw = np.ones(len(labels))
            for idx, m in enumerate(np.bincount(labels)):            
                lw[labels == idx] *= (m/float(labels.shape[0]))
            print accuracy_score(labels, classes, sample_weight = lw)
            fxs.append(fx) 
        return np.array(fxs).T                
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

#classify fx of the input layers
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
        self._S = S
        self._w, self._a, self._bias = self.fit_model(self._S, lab, "poly", "linear", 10, 0.01, 0.5)
        self._labels = self.predictions(self._S)
        print self._labels
        self._a = -self._w[0] / self._w[1] 
        self._weighted_labels = np.ones(lab.shape[0])
        from sklearn.metrics import accuracy_score
        for idx, m in enumerate(np.bincount(lab)):            
                self._weighted_labels[lab == idx] *= (m/float(lab.shape[0]))
        print accuracy_score(lab, self._labels, sample_weight = self._weighted_labels)                                         
        #calculate the parallels of separation 
        self._v = self._S * self._w + self._bias                                    
        self._yy = -self._w[0] * self._S - self._bias + self._v / self._w[1]                      
    def plot_emotion_classification(self):
	"""
	3D plotting of the classfication
	"""
            #3D plotting                                 
        from mpl_toolkits.mplot3d import Axes3D 
               
        fig = plt.figure()                                                                                                                    
        ax = Axes3D(fig)                                                    
        ax.plot3D(self._S, self._yy, 'k-')                                                       
        ax.scatter3D(self._S[:, 0], self._S[:, 1], c=self._labels, cmap=plt.cm.Paired)
       
        print (Fore.WHITE + "El grupo negativo '0' esta coloreado en azul, el grupo positivo '1' esta coloreado en rojo") 
                                                                      
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
        print (repr(negative_emotion_files)+"By arousal, emotion is sad and relaxed, not happy and not angry")

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
            fscaled = feature_scaling(features)
            labels = KMeans_clusters(fscaled)
            fx = svm_layers().layer_computation(fscaled, labels)
            sum_of_distances = np.sum(fx, axis=1)
            labl = np.sign(sum_of_distances)
            labl[labl==-1]=0
            print labl
            fx = np.vstack((sum_of_distances, np.ones(len(sum_of_distances)))).T #add ones to the distances as features
            msvm = main_svm(fx,np.int32(labl))
            msvm.plot_emotion_classification()
            neg_and_pos = msvm.neg_and_pos(files)
            emotions_data_dir()
            multitag_emotions_dir(tags_dirs, neg_and_pos[0], neg_and_pos[1], neg_arous_dir, pos_arous_dir)
        if sys.argv[2] in ('None'):
            files = descriptors_and_keys(files_dir, None)._files
            features = descriptors_and_keys(files_dir, None)._features
            fscaled = feature_scaling(features_combinations(features)._random_pairs)
            labels = KMeans_clusters(fscaled)
            labels_to_file = svm_layers().layer_computation(fscaled, labels, "poly")
            S = svm_layers().sum_of_S(fscaled)
            labl = svm_layers().best_labels(labels_to_file)
            print labl
            main_svm(fx,lab).plot_emotion_classification()
            neg_and_pos = main_svm(fx,lab).neg_and_pos(files)
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
