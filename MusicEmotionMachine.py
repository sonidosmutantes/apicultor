#! /usr/bin/env python
# -*- coding: utf-8 -*-

from SoundSimilarity import get_files, get_dics, desc_pair
from cross_validation import *
from DoSegmentation import *
import time
from colorama import Fore
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys                                                           
from essentia.standard import *
from smst.utils.audio import write_wav    
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer as lab_bin
from sklearn.preprocessing import LabelEncoder as le
from sklearn.decomposition.pca import PCA
from sklearn.cluster import KMeans
from random import *
from sklearn.utils.extmath import safe_sparse_dot as ssd
import librosa
from librosa import *                                       
import shutil
from smst.models import stft 
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   

# a memoize class for faster computing
class memoize:       
    """
    The memoize class stores returned results into a cache so that those can be used later on if the same case happens
    """                    
    def __init__(self, func, size = 512):
	"""
	memoize class init
	:param func: the function that you're going to decorate
	:param size: your maximum cache size (in MB)
	"""     
        self.func = func
        self.known_keys = [] #a list of keys to save numpy arrays
        self.known_values = [] #a list of values to save numpy results
        self.size = int(size * 1e+6 if size else None) #size in bytes
        self.size_copy = int(np.copy(self.size))

    def __call__(self, *args, **kwargs):
        key = hash(str(args))
        if (np.array(key) in np.array(self.known_keys)) == False: #when an ammount of arguments can't be identified from keys
            value = self.func(*args, **kwargs) #compute function
            self.known_keys.append(key) #add the key to your list of keys
            self.known_values.append(value) #add the value to your list of values
            if self.size is not None:
                self.size_copy -= sys.getsizeof(value) #the assigned space has decreased
                if (sys.getsizeof(self.known_values) > self.size_copy): #free space when size of values goes beyond the size limit
                    self.size_copy += sys.getsizeof(self.known_values[0:3]) #recover some size limit          
                    del self.known_keys[0:3] #delete the first three keys 
                    del self.known_values[0:3] #delete the first three values
            return value
        else: #if you've already computed everything
            i = self.known_keys.index(key) #find your key and return your values
            return self.known_values[i]

                            

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
      - scaled features with mean and standard deviation
    """    
    mu = np.mean(f)
    sigma = np.std(f)
    return (f - mu) / sigma

def KMeans_clusters(fscaled):
    """
    KMeans clustering for features                                                           
    :param fscaled: scaled features                                                                                         
    :returns:                                                                                                         
      - labels: classes         
    """
    labels = (KMeans(init = PCA(n_components = 4).fit(fscaled).components_, n_clusters=4, n_init=1, precompute_distances = True).fit(fscaled).labels_)
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
    @memoize
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
    @memoize
    def linear_kernel_matrix(self, x, y): 
	"""
	Custom inner product. The returned matrix is a Linear Kernel Gram Matrix
	:param x: your dataset
	:param y: another dataset (sth like transposed(x))
	"""       
        return np.dot(x,y)
    @classmethod
    @memoize
    def sigmoid_kernel(self, x, y, gamma):
	"""
	Custom sigmoid kernel function, similarities of vectors in a sigmoid kernel matrix
	:param x: array of input vectors
	:param y: array of input vectors
	:param gamma: gamma
	:returns:
	  - sk: inner product
	"""
        c = 1

        sk = ssd(x, y.T, dense_output = True)

        sk *= gamma

        sk += c

        np.tanh(np.array(sk, dtype='float64'), np.array(sk, dtype = 'float64'))
        return np.array(sk, dtype = 'float64')
    @classmethod
    @memoize
    def rbf_kernel(self, x, y, gamma):
	"""
	Custom sigmoid kernel function, similarities of vectors using a radial basis function kernel
	:param x: array of input vectors
	:param y: array of input vectors
	:param gamma: reach factor
	:returns:
	  - rbfk: radial basis of the kernel's inner product
	"""     
        mat1 = np.mat(x) #convert to readable matrices
        mat2 = np.mat(y)                                                                                                
        trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T #norm matrices
        trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat2]).T                                                                                
        k1 = trnorms1 * np.mat(np.ones((mat2.shape[0], 1), dtype=np.float64)).T #dot products of y and y transposed and x and x transposed   
        k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T          
                   
        rbfk = k1 + k2 #sum products together
        rbfk -= 2 * np.mat(mat1 * mat2.T) #dot product of x and y transposed                                         
        rbfk *= - 1./(2 * np.power(gamma, 2)) #radial basis
        np.exp(rbfk,rbfk)
        return np.array(rbfk)

    def fit_model(self, features,labels, kernel1, kernel2, C, reg_param, gamma):
	"""
	Fit your data for classification. Attributes such as alpha, the bias, the weight, and the support vectors can be accessed after training.
	:param features: your features dataset
	:param labels: your labels dataset
	:param kernel1: the kernel used for predictive task
	:param kernel2: the kernel used to solve the min-max problem
	:param C: cost, also called penalty parameter
	:param reg_param: tolerance, used for convergence
	:param gamma: determines a reach 
	*What you can have access to after computing
	  - w: the weight coefficient
	  - a: resolutions to min-max problems
	  - dual_coefficients: stacked alphas of the support vectors appearing in all of the problems of its class
	  - svs: find out which index of your features is a support vector
	  - nvs: number of support vectors in each class
	  - ns: where your support vectors are from your input dataset
	  - bias: a helper
	"""    
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.gamma = gamma
        self.C = C

        features = np.ascontiguousarray(features)

        multiclass = False

        self.targets = labels

        self.n_class = len(np.bincount(self.targets))

        if self.n_class == 1:                                      
            raise Exception("There has to be two or more targets")

        if reg_param == 0:                                      
            raise ValueError("There's no such 0 tolerance")

        if self.n_class > 2:                                      
            multiclass = True
        if not self.n_class > 2:                                      
            raise Exception("Two labels classification for music is not allowed here")

        if multiclass == True:
            self.instances = list(combinations(range(self.n_class), 2))
            self.classes = np.unique(self.targets)
            self.classifications = len(self.instances)
            self.classesm = defaultdict(list) 
            self.trained_features = defaultdict(list) 
            self.indices_features = defaultdict(list) 
            self.a = defaultdict(list)
            self.bias = defaultdict(list)
            self.pos_alphas = defaultdict(list)
            self.w = np.zeros((self.classifications, features.shape[1]))
            self.svs = defaultdict(list)
            self.ns = defaultdict(list)
            self.nvs = defaultdict(list)
            self.dual_coefficients = [[] for i in range(self.n_class - 1)]

        else: 
            self.instances = [(0,1)]
            self.classes = np.unique(self.targets)                                
            self.classifications = 1
            self.classesm = defaultdict(list)
            self.trained_features = defaultdict(list) 
            self.indices_features = defaultdict(list)  
            self.a = defaultdict(list)
            self.bias = defaultdict(list)
            self.pos_alphas = defaultdict(list)
            self.w = np.zeros(features.shape[1])
            self.svs = defaultdict(list)
            self.ns = defaultdict(list)
            self.nvs = defaultdict(list)
            self.dual_coefficients = defaultdict(list) 

        for c in range(self.classifications):   

            self.classesm[c] = np.concatenate((self.targets[self.targets==self.instances[c][0]],self.targets[self.targets==self.instances[c][1]]))
            self.indices_features[c] = np.concatenate((np.where(self.targets==self.instances[c][0]),np.where(self.targets==self.instances[c][1])), axis = 1)[0]
            self.trained_features[c] = np.ascontiguousarray(np.concatenate((features[self.targets==self.instances[c][0]],features[self.targets==self.instances[c][1]])))

            self.classesm[c] = lab_bin(0,1).fit_transform(self.classesm[c]).T[0]

            self.trained_features[c].flags.writeable = False 
                                                                     
            lab = self.classesm[c] * 2 - 1

            x = np.asanyarray([np.hstack((self.trained_features[c][i],lab[i])) for i in range(len(self.trained_features[c]))]) 
                                                                   
            if self.kernel2 == "linear":                                                                      
                kernel = self.linear_kernel_matrix
                Qvalue = kernel(self.trained_features[c], self.trained_features[c].T)            
            if self.kernel2 == "poly":                                
                kernel = self.polynomial_kernel
                Qvalue = kernel(self.trained_features[c], self.trained_features[c], gamma)         
            if self.kernel2 == "rbf":                                                                      
                kernel = self.rbf_kernel
                Qvalue = kernel(self.trained_features[c], self.trained_features[c], gamma)               
            if self.kernel2 == "sigmoid":                             
                kernel = self.sigmoid_kernel 
                Qvalue = kernel(self.trained_features[c], self.trained_features[c], gamma)           
            if self.kernel2 == None:                                  
                print ("Apply a kernel")    
                                                                  
            Q = Qvalue * lab                   

            matrix = np.asanyarray([np.hstack((Q[i],1)) for i in range(len(Q))])
                                                                      
            class_weight = float(len(self.trained_features[c]))/(2 * np.bincount(self.classesm[c]))

            a = np.zeros(len(self.trained_features[c]))                     
                           
            iterations = 0                                            
            diff = 1. + reg_param 
            for i in range(len(self.trained_features[c])):  
                data_i = range(len(self.trained_features[c]))
                C = self.C                                             
                for k in range(1, len(self.trained_features[c])):                
                    a0 = a.copy()                                     
                    try:                                              
                        p1 = k                                        
                        p2 = k
                        loc0 = data_i[:len(features[self.targets==self.instances[c][0]])]                                     
                        loc1 = data_i[len(features[self.targets==self.instances[c][0]])+1:]                                        
                        while (p1 == k) and (p2 == k):                                                                      
                            p1 = random.randint(min(loc0), max(loc0))
                            p2 = random.randint(min(loc1), max(loc1))
                    except Exception, e:                              
                        break                                         
                    g1 = 1 - lab[p1] * np.sum(a0[p1] * lab[p1] * Q[p1,p1])
                    g2 = 1 - lab[p1] * np.sum(a0[p1] * lab[p1] * Q[p2,p2])                            
                    direction_grad1 = 1./np.diag(Q)[p1] * (class_weight[self.classesm[c][p1]] * (g1 + (1 - class_weight[self.classesm[c][p1]]) * g2))
                    direction_grad2 = 1./np.diag(Q)[p2] * (class_weight[self.classesm[c][p2]] * (g1 + (1 - class_weight[self.classesm[c][p2]]) * g2))
                    if iterations > 14 and ((direction_grad1 * (g1 + g2) <= 0) or (direction_grad2 * (g1 + g2) <= 0)):        
                        break
                    a[p1] = a0[p1] + direction_grad1
                    a[p2] = a0[p2] + direction_grad2
                    if a[p1] < 0.0:
                        a[p1] = 0.0
                    if a[p2] < 0.0:
                        a[p2] = 0.0
                    C1 = [self.C / p1 if p1 != 0 else self.C][0]
                    C2 = [self.C / p2 if p2 != 0 else self.C][0]
                    a[p1] = min(min(a[p1], C * class_weight[self.classesm[c][p1]] * class_weight[self.classesm[c][p1]]), np.sum((a[p1]*lab[p1])**2))
                    a[p2] = min(min(a[p2], C * class_weight[self.classesm[c][p2]] * class_weight[self.classesm[c][p2]]), np.sum((a[p1]*lab[p1])**2))
                    try:                   
                        del data_i[data_i[np.where(p1 == np.array(p1))[0]]]
                        del data_i[data_i[np.where(p2 == np.array(p2))[0]]]
                    except Exception, e:
                        break 
                diff = sum(a-a0)
                if iterations > 14 and (diff > -reg_param and diff < reg_param):
                    break
                iterations += 1

            print ("Iterated " + str(iterations) +  " times for gradient ascent")                                                           

            #taken from Pegasos algorithm by avaitla, alphas are automatically signed and those with no incidence in the hyperplane are 0 complex or -0 complex, so alphas != 0 are taken 
            iterations = 1
            Q = Q * lab
            for i in range(14):
                for tau in range(len(a)):
                    wx = np.dot(a, Q[tau,:])
                    a *= ((1-1./iterations))
                    if(lab[tau]*wx < 1):
                        a[tau] = float(lab[tau])/float(reg_param * iterations)
                iterations += 1
                               
            print ("Iterated " + str(iterations - 1) +  " times for stochastic gradient descent")                                    

            self.ns[c].append([])
            self.ns[c].append([])

            for dc in range(len(a)):
                if a[dc] != 0.0:
                    if a[dc] > 0.0:
                        self.ns[c][1].append(self.indices_features[c][dc])
                    else:
                        self.ns[c][0].append(self.indices_features[c][dc])
                else:
                    pass

            self.svs[c] = self.trained_features[c][a != 0.0, :]  
                                                                      
            self.a[c] = a[a != 0.0]    

            self.bias[c] = np.zeros((1, len(self.svs[c][0]))) #bias

            for j in xrange(len(self.svs[c])):
                if self.kernel1 == "sigmoid":     
                    self.bias[c] += self.a[c][j] * self.sigmoid_kernel(self.svs[c][j,:],self.svs[c][0,:], self.gamma)
                if self.kernel1 == "poly":                                                                     
                    self.bias[c] += self.a[c][j] * self.polynomial_kernel(self.svs[c][j,:],self.svs[c][0,:], self.gamma)          
                if self.kernel1 == "rbf":                        
                    self.bias[c] += self.a[c][j] * self.rbf_kernel(np.float64(self.svs[c][j,:]),np.float64(self.svs[c][0,:]), self.gamma)
                if self.kernel1 == "linear":                     
                    self.bias[c] += self.a[c][j] * self.linear_kernel_matrix(self.svs[c][j,:],self.svs[c][0,:]) 

            self.bias[c] = self.bias[c][0][0]

        a = defaultdict(list)
        for i, (j,m) in enumerate(self.instances):
            a[j].append(self.a[i][self.a[i] < 0])
            a[m].append(self.a[i][self.a[i] > 0])

        ns = defaultdict(list)
        for i, (j,m) in enumerate(self.instances):
            ns[j].append(self.ns[i][0])
            ns[m].append(self.ns[i][1])

        nsv = [np.hstack(ns.values()[j][i] for i in range(len(ns.values()[j]))) for j in range(len(ns.values()))]

        unqs = np.array([np.unique(nsv[i], return_index = True, return_counts = True) for i in range(len(nsv))])

        uniques = [list(unqs[:,0][i]) for i in range(len(unqs[:,0]))]

        counts = [list(unqs[:,2][i]) for i in range(len(unqs[:,0]))]

        svs_ = [list(np.array(counts[i]) == self.n_class - 1) for i in range(len(counts))]

        self.ns = [uniques[i][j] for i in range(len(uniques)) for j in range(len(uniques[i])) if counts[i][j] >= self.n_class - 1]

        for i in range(len(a)):           
            for j in range(len(a[i])):
                for m in range(len(a[i][j])):
                    if ns[i][j][m] in self.ns:
                        self.dual_coefficients[j].append(a[i][j][m])

        self.dual_coefficients = np.array(self.dual_coefficients)

        self.nvs = [len(np.array(uniques[i])[np.array(svs_[i])]) for i in range(len(uniques))]

        self.svs = features[self.ns] #update support vectors         

        self.bias = np.array([self.bias[i] for i,x in enumerate(self.bias)])

        sv_locs = np.cumsum(np.hstack([[0], self.nvs]))

        self.w = [] #update weights given common support vectors, the other values helped making sure it wasn't restarting

        for class1 in range(self.n_class):
            # SVs for class1:
            sv1 = self.svs[sv_locs[class1]:sv_locs[class1 + 1], :]
            for class2 in range(class1 + 1, self.n_class):
                # SVs for class1:
                sv2 = self.svs[sv_locs[class2]:sv_locs[class2 + 1], :]
                # dual coef for class1 SVs:
                alpha1 = self.dual_coefficients[class2 - 1, sv_locs[class1]:sv_locs[class1 + 1]]
                # dual coef for class2 SVs:
                alpha2 = self.dual_coefficients[class1, sv_locs[class2]:sv_locs[class2 + 1]]
                # build weight for class1 vs class2
                self.w.append(ssd(alpha1, sv1)
                            + ssd(alpha2, sv2))

        return self 

    def decision_function(self, features):
	"""
	Compute the distances from the separating hyperplane
	:param features: your features dataset
	:returns:
	  - decision function with added bias
	""" 

        if len(np.float64(features).shape) != len(self.svs.shape):
            features = np.ascontiguousarray(np.resize(features, (len(features), self.svs.shape[1])))
        else:
            features = np.ascontiguousarray(features)

        if self.kernel1 == "poly":
            kernel = self.polynomial_kernel
        if self.kernel1 == "sigmoid":
            kernel = self.sigmoid_kernel
        if self.kernel1 == "rbf":
            kernel = self.rbf_kernel 
        if self.kernel1 == "linear":
            kernel = self.linear_kernel_matrix
                                               
        if self.kernel1 == "linear":           
            k = kernel(self.svs, features.T)       
        else:                                             
            k = kernel(self.svs, features, self.gamma)     

        start = [sum(self.nvs[:i]) for i in range(len(self.nvs))]
        end = [start[i] + self.nvs[i] for i in range(len(self.nvs))]
        c = [ sum(self.dual_coefficients[ i ][p] * k[p] for p in range(start[j], end[j])) +
              sum(self.dual_coefficients[j-1][p] * k[p] for p in range(start[i], end[i]))
                for i in range(len(self.nvs)) for j in range(i+1,len(self.nvs))]

        dec = np.array([sum(x) for x in zip(c, self.bias)]).T

        #one vs rest based on scikit-learn

        predictions = dec < 0 #biased? predictions again

        confidences = dec * -1 #variables don't really matter this time

        votes = np.zeros((len(features), self.n_class))                      
        sum_of_confidences = np.zeros((len(features), self.n_class))                                         
        K = 0                                         
        for i in range(self.n_class):
            for j in range(i + 1,self.n_class):
                sum_of_confidences[:,i] -= confidences[:,K]
                sum_of_confidences[:,j] += confidences[:,K]
                votes[predictions[:,K] == 0, i] += 1
                votes[predictions[:,K] == 1, j] += 1
                K += 1

        #some bounds to decide how certain is the function
        max_confi = sum_of_confidences.max()
        min_confi = sum_of_confidences.min()

        if max_confi == min_confi:            
            return votes  

        eps = np.finfo(sum_of_confidences.dtype).eps
        max_abs_confi = max(abs(max_confi), abs(min_confi)) #maximum absolute confidence to set up a max scaling value
        scale = (0.5 - eps) / max_abs_confi #scale everything off and break ties in voting, no decision is being switched
        #scale
        return votes + sum_of_confidences * scale

    def predictions(self, features, targts): 
	"""
	Prediction output
	:param features: your features dataset
	:param targts: your targets dataset
	:returns:
	  - __these_predicted: predicted targets
	"""        

        decision = self.decision_function(features)
        
        self.__these_predicted = np.copy(targts)

        votes = np.zeros((len(features), self.n_class))

        for i in self.classes:
            for j in range(i + 1, len(self.classes)):
                votes[:,i][decision[:,i] > 0] += 1
                votes[:,j][decision[:,j] < 0] += 1 

        for  v in range(votes.shape[0]):
             where_max_votes = np.where(votes[v] == votes[v].max())   
                                          
             if len(list(where_max_votes)[0]) > 1: #if there's a tie it won't be broken in prediction and the feature has its past target
                    pass
             else:                       
                    self.__these_predicted[v] = list(where_max_votes)[0] #assign the most voted class to a feature

        return self.__these_predicted

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
        self.fxs = []
        self.targets_outputs = []
        sample_weight = float(len(features)) / (len(np.array(list(set(labels)))) * np.bincount(labels))
        Cs = [1./0.01, 1./0.33, 1./0.04, 1./0.06, 1./0.08, 1./0.1, 1./0.2, 1./0.4, 1./0.6, 1./0.8, 1./2, 1./20]
        reg_params = [0.01, 0.33, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 2, 20]
        for i in range(0,12):
            print ("Calculating values for prediction")
            if i == 0:
                best_estimator0 = GridSearch(self, features, labels, Cs[:2], reg_params[:2], [['rbf', 'sigmoid']])
                print ("Best estimators are: ") + str(best_estimator0['C'][0]) + " for and " + str(best_estimator0['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "rbf", "sigmoid", best_estimator0['C'][0][0], best_estimator0['reg_param'][0][0], 1./features[labels==0].shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c0 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c0)
                print ("Predicted"), pred_c0
                print score(labels, pred_c0)
            if i == 1:
                best_estimator1 = GridSearch(self, features, labels, Cs[:2], reg_params[:2], [['poly', 'rbf']])
                print ("Best estimators are: ") + str(best_estimator1['C'][0]) + " for and " + str(best_estimator1['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "poly", "rbf", best_estimator1['C'][0][0], best_estimator1['reg_param'][0][0], 1./features[labels==1].shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c1 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c1)
                print ("Predicted"), pred_c1
                print score(labels, pred_c1)
            if i == 2:
                best_estimator2 = GridSearch(self, features, labels, Cs, reg_params, [['poly', 'linear']])
                print ("Best estimators are: ")  + str(best_estimator2['C'][0]) + " for and " + str(best_estimator2['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "poly", "linear", best_estimator2['C'][0][0], best_estimator2['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c2 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c2)
                print ("Predicted"), pred_c2
                print score(labels, pred_c2)
            if i == 4:
                best_estimator4 = GridSearch(self, features, labels, Cs, reg_params, [['rbf', 'poly']])
                print ("Best estimators are: ") + str(best_estimator4['C'][0]) + " for and " + str(best_estimator4['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "rbf", "poly", best_estimator4['C'][0][0], best_estimator4['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c4 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c4)
                print ("Predicted"), pred_c4
                print score(labels, pred_c4)
            if i == 5:
                best_estimator5 = GridSearch(self, features, labels, Cs, reg_params, [['rbf', 'linear']])
                print ("Best estimators are: ") + str(best_estimator5['C'][0]) + " for and " + str(best_estimator5['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "rbf", "linear", best_estimator5['C'][0][0], best_estimator5['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c5 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c5)
                print ("Predicted"), pred_c5
                print score(labels, pred_c5)
            if i == 6:
                best_estimator6 = GridSearch(self, features, labels, Cs, reg_params, [['linear', 'rbf']])
                print ("Best estimators are: ") + str(best_estimator6['C'][0]) + " for and " + str(best_estimator6['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "linear", "rbf", best_estimator6['C'][0][0], best_estimator6['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c6 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c6)
                print ("Predicted"), pred_c6
                print score(labels, pred_c6)
            if i == 7:
                best_estimator7 = GridSearch(self, features, labels, Cs, reg_params, [['sigmoid', 'rbf']])
                print ("Best estimators are: ") + str(best_estimator7['C'][0]) + " for and " + str(best_estimator7['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "sigmoid", "rbf", best_estimator7['C'][0][0], best_estimator7['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c7 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c7)
                print ("Predicted"), pred_c7
                print score(labels, pred_c7)
            if i == 8:
                best_estimator8 = GridSearch(self, features, labels, Cs, reg_params, [['sigmoid', 'poly']])
                print ("Best estimators are: ") + str(best_estimator8['C'][0]) + " for and " + str(best_estimator8['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "sigmoid", "poly", best_estimator8['C'][0][0], best_estimator8['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c8 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c8)
                print ("Predicted"), pred_c8
                print score(labels, pred_c8)
            if i == 9:
                best_estimator9 = GridSearch(self, features, labels, Cs, reg_params, [['poly', 'sigmoid']])
                print ("Best estimators are: ") + str(best_estimator9['C'][0]) + " for and " + str(best_estimator9['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "poly", "sigmoid", best_estimator9['C'][0][0], best_estimator9['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c9 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c9)
                print ("Predicted"), pred_c9
                print score(labels, pred_c9)
            if i == 10:
                best_estimator10 = GridSearch(self, features, labels, Cs, reg_params, [['sigmoid', 'linear']])
                print ("Best estimators are: ") + str(best_estimator10['C'][0]) + " for and " + str(best_estimator10['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "sigmoid", "linear", best_estimator10['C'][0][0], best_estimator10['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c10 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c10)
                print ("Predicted"), pred_c10
                print score(labels, pred_c10)
            if i == 11:
                best_estimator11 = GridSearch(self, features, labels, Cs, reg_params, [['linear', 'sigmoid']])
                print ("Best estimators are: ") + str(best_estimator11['C'][0]) + " for and " + str(best_estimator11['reg_param'][0]) + " for regularization parameter"
                self.fit_model(features, labels, "linear", "sigmoid", best_estimator11['C'][0][0], best_estimator11['reg_param'][0][0], 1./features.shape[1])
                print ("Predicting")
                fx = self.decision_function(features)
                pred_c11 = self.predictions(features, labels)
                self.targets_outputs.append(pred_c11)
                print ("Predicted"), pred_c11
                print score(labels, pred_c11)
            self.fxs.append(fx)
  
        return self

    def sum_fx(self):
	"""
	Sum all distances along the first axis
	"""
        return np.array(self.fxs).sum(axis=0) 

    def best_labels(self):
	"""
	Get the labels with highest output in the input layers instances
	"""
        self.targets_outputs = np.int32([Counter(np.int32(self.targets_outputs)[:,i]).most_common()[0][0] for i in range(np.array(self.targets_outputs).shape[1])])

        return self.targets_outputs 

def best_kernels_output(best_estimator, kernel_configs):
    """
    From a list of best configurations, get the highest configuration (to find out which kernels you should use)                                                           
    :param best_estimator: a cross-validation output of best estimators of various multi-kernel analysis    
    :param kernel_configs: your kernels selections that you used for cross-validation                                                                                        
    :returns:                                                                                                         
      - C: best C 
      - reg_param: best reg_param 
      - kernel_conf: best configuration of kernels to use         
    """  
    max_score = np.array(best_estimator['score'][0]).argmax()    
    C = best_estimator['C'][max_score]    
    reg_param = best_estimator['reg_param'][max_score]       
    kernel_conf = kernel_configs[max_score]    
    return C, reg_param, kernel_conf 

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
    def __init__(self, S, lab, C, reg_param, gamma, kernels_config):
        super(deep_support_vector_machines, self).__init__()
        self._S = S
        self.fit_model(self._S, lab, kernels_config[0], kernels_config[1], C, reg_param, gamma)
        self._labels = self.predictions(self._S, lab)
        print self._labels
        print score(lab, self._labels)
                                                    
    def neg_and_pos(self, files):
	"""
	Lists of files according to emotion label (0 for negative, 1 for positive)
	:param files: filenames of inputs
	:returns:
	  - negative_emotion_files: files with negative emotional value (emotional meaning according to the whole performance)
	  - positive_emotion_files: files with positive emotional value (emotional meaning according to the whole performance) 
	"""
        angry_group = map(lambda json: files[json], [i for i, x in enumerate(self._labels) if x ==0])
        self.angry_files = [i.split('.json')[0] for i in angry_group]
        relaxed_group = map(lambda json: files[json], [i for i, x in enumerate(self._labels) if x ==2]) 
        self.relaxed_files = [i.split('.json')[0] for i in relaxed_group]
        sad_group = map(lambda json: files[json], [i for i, x in enumerate(self._labels) if x ==1])
        self.sad_files = [i.split('.json')[0] for i in sad_group]
        happy_group = map(lambda json: files[json], [i for i, x in enumerate(self._labels) if x ==3]) 
        self.happy_files = [i.split('.json')[0] for i in happy_group]
        return self

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
    if not os.path.exists(files_dir+'/emotions/remixes/happy'):
        os.makedirs(files_dir+'/emotions/remixes/happy')
    if not os.path.exists(files_dir+'/emotions/remixes/sad'):
        os.makedirs(files_dir+'/emotions/remixes/sad')
    if not os.path.exists(files_dir+'/emotions/remixes/angry'):
        os.makedirs(files_dir+'/emotions/remixes/angry')
    if not os.path.exists(files_dir+'/emotions/remixes/relaxed'):
        os.makedirs(files_dir+'/emotions/remixes/relaxed')
    if not os.path.exists(files_dir+'/emotions/remixes/not happy'):
        os.makedirs(files_dir+'/emotions/remixes/not happy')
    if not os.path.exists(files_dir+'/emotions/remixes/not sad'):
        os.makedirs(files_dir+'/emotions/remixes/not sad')
    if not os.path.exists(files_dir+'/emotions/remixes/not angry'):
        os.makedirs(files_dir+'/emotions/remixes/not angry')
    if not os.path.exists(files_dir+'/emotions/remixes/not relaxed'):
        os.makedirs(files_dir+'/emotions/remixes/not relaxed')

#look for all downloaded audio
tags_dirs = lambda files_dir: [os.path.join(files_dir,dirs) for dirs in next(os.walk(os.path.abspath(files_dir)))[1]]

#emotions dictionary directory (to use with RedPanal API)
def multitag_emotions_dictionary_dir():
    """                                                                                     
    create emotions dictionary directory                                        
    """           
    os.makedirs('data/emotions_dictionary')

# save files in tag directory according to emotion using hpss
def bpm_emotions_remix(files_dir, sad_files, happy_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param sad_files: your list of sad files                                   
    :param happy_files: your list of happy files                                                                                   
    :param files_dir: data tag dir                                           
    """                                                                                         

    if happy_files:
        print (repr(happy_files)+"emotion is happy")

    if sad_files:
        print (repr(sad_files)+"emotion is sad")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(sad_files)
    files_2 = set(sound_names).intersection(happy_files)
                                                           
    try:                                                      
        if files_1 and (not os.path.exists(files_dir+'/tempo/sad')):                                                    
            os.mkdir(files_dir+'/tempo/sad')                                                              
        if files_2 and (not os.path.exists(files_dir+'/tempo/happy')):                                               
            os.mkdir(files_dir+'/tempo/happy')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                      
                                        
    for e in files_1:
        shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.ogg', files_dir+'/tempo/sad/'+(str(e))+'tempo.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/tempo/'+(str(e))+'tempo.ogg', files_dir+'/tempo/happy/'+(str(e))+'tempo.ogg')


    happiness_dir = files_dir+'/tempo/happy' 
    for subdirs, dirs, sounds in os.walk(happiness_dir):  
    	happy_audio = [MonoLoader(filename=happiness_dir+'/'+happy_f)() for happy_f in sounds]
    happy_audio = [scratch_music(i) for i in happy_audio]
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
    sad_audio = [scratch_music(i) for i in sad_audio]
    sad_N = min([len(i) for i in sad_audio])  
    sad_samples = [i[:sad_N]/i.max() for i in sad_audio]  
    sad_x = np.array(sad_samples).sum(axis=0) 
    sad_X = 0.5*sad_x/sad_x.max()
    sad_Harmonic, sad_Percussive = decompose.hpss(librosa.core.stft(sad_X))
    sad_harmonic = istft(sad_Harmonic)  
    MonoWriter(filename=files_dir+'/tempo/sad/'+'sad_mix_bpm.ogg', format = 'ogg', sampleRate = 44100)(sad_harmonic) 

def attack_emotions_remix(files_dir, relaxed_files, angry_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param relaxed_files: your relaxed files                                    
    :param angry_files: your angry files                                                                                  
    :param files_dir: data tag dir                                           
    """                                                                                         

    if angry_files:
        print (repr(angry_files)+"emotion is angry")

    if relaxed_files:
        print (repr(relaxed_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(angry_files)
    files_2 = set(sound_names).intersection(relaxed_files)
                                                           
    try:                                                      
        if files_1 and (not os.path.exists(files_dir+'/attack/angry')):                                                    
            os.mkdir(files_dir+'/attack/angry')                                                              
        if files_2 and (not os.path.exists(files_dir+'/attack/relaxed')):                                               
            os.mkdir(files_dir+'/attack/relaxed')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                         
                                        
    for e in files_1:
        shutil.copy(files_dir+'/attack/'+(str(e))+'attack.ogg', files_dir+'/attack/angry/'+(str(e))+'attack.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/attack/'+(str(e))+'attack.ogg', files_dir+'/attack/relaxed/'+(str(e))+'attack.ogg')


    anger_dir = files_dir+'/attack/angry' 
    for subdirs, dirs, sounds in os.walk(anger_dir):  
    	angry_audio = [MonoLoader(filename=anger_dir+'/'+angry_f)() for angry_f in sounds]
    angry_audio = [scratch_music(i) for i in angry_audio]
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
    tender_audio = [scratch_music(i) for i in tender_audio]
    tender_N = min([len(i) for i in tender_audio])  
    tender_samples = [i[:tender_N]/i.max() for i in tender_audio]  
    tender_x = np.array(tender_samples).sum(axis=0) 
    tender_X = 0.5*tender_x/tender_x.max()
    tender_Harmonic, tender_Percussive = decompose.hpss(librosa.core.stft(tender_X))
    tender_harmonic = istft(tender_Harmonic)  
    MonoWriter(filename=files_dir+'/attack/relaxed/relaxed_mix_attack.ogg', format = 'ogg', sampleRate = 44100)(tender_harmonic) 

def dissonance_emotions_remix(files_dir, relaxed_files, angry_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param relaxed_files: your relaxed files                                   
    :param angry_files: your angry files                                                                                
    :param files_dir: data tag dir                                           
    """                                                                                         

    if angry_files:
        print (repr(angry_files)+"emotion is angry")

    if relaxed_files:
        print (repr(relaxed_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(angry_files)
    files_2 = set(sound_names).intersection(relaxed_files)
                                                           
    try:                                                      
        if files_1 and (not os.path.exists(files_dir+'/dissonance/sad')) :                                                    
            os.mkdir(files_dir+'/dissonance/sad')                                                              
        if files_2 and (not os.path.exists(files_dir+'/dissonance/happy')):                                               
            os.mkdir(files_dir+'/dissonance/happy')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                         
                                        
    for e in files_1:
        shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.ogg', files_dir+'/dissonance/angry/'+(str(e))+'dissonance.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/dissonance/'+(str(e))+'dissonance.ogg', files_dir+'/dissonance/relaxed/'+(str(e))+'dissonance.ogg')


    fear_dir = files_dir+'/dissonance/angry' 
    for subdirs, dirs, sounds in os.walk(fear_dir):  
    	fear_audio = [MonoLoader(filename=fear_dir+'/'+fear_f)() for fear_f in sounds]
    fear_audio = [scratch_music(i) for i in fear_audio]
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
    happy_audio = [scratch_music(i) for i in happy_audio]
    happy_N = min([len(i) for i in happy_audio])  
    happy_samples = [i[:happy_N]/i.max() for i in happy_audio]  
    happy_x = np.array(happy_samples).sum(axis=0) 
    happy_X = 0.5*happy_x/happy_x.max()
    happy_Harmonic, happy_Percussive = decompose.hpss(librosa.core.stft(happy_X))
    happy_harmonic = istft(happy_Harmonic)  
    MonoWriter(filename=files_dir+'/dissonance/relaxed/relaxed_mix_dissonance.ogg', format = 'ogg', sampleRate = 44100)(happy_harmonic) 

def mfcc_emotions_remix(files_dir, relaxed_files, angry_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param relaxed_files: your relaxed files                                   
    :paramangry_files: your angry files                                                                                  
    :param files_dir: data tag dir                                           
    """                                                                                         

    if angry_files:
        print (repr(angry_files)+"emotion is angry")

    if relaxed_files:
        print (repr(relaxed_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(angry_files)
    files_2 = set(sound_names).intersection(relaxed_files)
                                                           
    try:                                                      
        if files_1 and (not os.path.exists(files_dir+'/mfcc/angry')):                                                    
            os.mkdir(files_dir+'/mfcc/angry')                                                              
        if files_2 and (not os.path.exists(files_dir+'/mfcc/relaxed')):                                               
            os.mkdir(files_dir+'/mfcc/relaxed')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.ogg', files_dir+'/mfcc/angry/'+(str(e))+'mfcc.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/mfcc/'+(str(e))+'mfcc.ogg', files_dir+'/mfcc/relaxed/'+(str(e))+'mfcc.ogg')


    fear_dir = files_dir+'/mfcc/angry' 
    for subdirs, dirs, sounds in os.walk(fear_dir):  
    	fear_audio = [MonoLoader(filename=fear_dir+'/'+fear_f)() for fear_f in sounds]
    fear_audio = [scratch_music(i) for i in fear_audio]
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
    happy_audio = [scratch_music(i) for i in happy_audio]
    happy_N = min([len(i) for i in happy_audio])  
    happy_samples = [i[:happy_N]/i.max() for i in happy_audio]  
    happy_x = np.array(happy_samples).sum(axis=0) 
    happy_X = 0.5*happy_x/happy_x.max()
    happy_Harmonic, happy_Percussive = decompose.hpss(librosa.core.stft(happy_X))
    happy_harmonic = istft(happy_Harmonic)  
    MonoWriter(filename=files_dir+'/mfcc/relaxed/relaxed_mix_mfcc.ogg', format = 'ogg', sampleRate = 44100)(happy_harmonic) 
 

def centroid_emotions_remix(files_dir, relaxed_files, angry_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param relaxed_files: your relaxed files                                    
    :param angry_files: your angry files                                                                                   
    :param files_dir: data tag dir                                          
    """                                                                                         

    if angry_files:
        print (repr(angry_files)+"emotion is angry")

    if relaxed_files:
        print (repr(relaxed_files)+"emotion is relaxed")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    files_1 = set(sound_names).intersection(angry_files)
    files_2 = set(sound_names).intersection(relaxed_files)
                                                           
    try:                                                      
        if files_1 and (not os.path.exists(files_dir+'/centroid/angry')):                                                    
            os.mkdir(files_dir+'/centroid/angry')                                                              
        if files_2 and (not os.path.exists(files_dir+'/centroid/relaxed')):                                               
            os.mkdir(files_dir+'/centroid/relaxed')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                       
                                        
    for e in files_1:
        shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.ogg', files_dir+'/centroid/angry/'+(str(e))+'centroid.ogg')
                                                                                         
    for e in files_2:
        shutil.copy(files_dir+'/centroid/'+(str(e))+'centroid.ogg', files_dir+'/centroid/relaxed/'+(str(e))+'centroid.ogg')


    fear_dir = files_dir+'/centroid/angry' 
    for subdirs, dirs, sounds in os.walk(fear_dir):  
    	fear_audio = [MonoLoader(filename=fear_dir+'/'+fear_f)() for fear_f in sounds]
    fear_audio = [scratch_music(i) for i in fear_audio]
    fear_N = min([len(i) for i in fear_audio])  
    fear_samples = [i[:fear_N]/i.max() for i in fear_audio]  
    fear_x = np.array(fear_samples).sum(axis=0) 
    fear_X = 0.5*fear_x/fear_x.max()
    fear_Harmonic, fear_Percussive = decompose.hpss(librosa.core.stft(fear_X))
    fear_harmonic = istft(fear_Harmonic) 
    MonoWriter(filename=files_dir+'/centroid/angry/angry_mix_centroid.ogg', format = 'ogg', sampleRate = 44100)(fear_harmonic)
  

    relaxed_dir = files_dir+'/centroid/relaxed'
    for subdirs, dirs, relaxed_sounds in os.walk(relaxed_dir):
    	relaxed_audio = [MonoLoader(filename=relaxed_dir+'/'+relaxed_f)() for relaxed_f in relaxed_sounds]
    relaxed_audio = [scratch_music(i) for i in relaxed_audio]
    relaxed_N = min([len(i) for i in relaxed_audio])  
    relaxed_samples = [i[:relaxed_N]/i.max() for i in relaxed_audio]  
    relaxed_x = np.array(relaxed_samples).sum(axis=0) 
    relaxed_X = 0.5*relaxed_x/relaxed_x.max()
    relaxed_Harmonic, relaxed_Percussive = decompose.hpss(librosa.core.stft(relaxed_X))
    relaxed_harmonic = istft(relaxed_Harmonic)  
    MonoWriter(filename=files_dir+'/centroid/relaxed/relaxed_mix_centroid.ogg', format = 'ogg', sampleRate = 44100)(relaxed_harmonic) 

def hfc_emotions_remix(files_dir, classes_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param classes_files: neg_and_pos result (read main_svm class documentation)
    :param files_dir: data tag dir                                           
    """                                                                                         

    if classes_files:
        print (repr([classes_files.happy_files, classes_files.angry_files, classes_files.relaxed_files])+"emotion is not sad")

    if classes_files.sad_files:
        print (repr(classes_files.sad_files)+"emotion is sad")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break   

    not_sad_f = list([set(classes_files.happy_files), set(classes_files.angry_files), set(classes_files.relaxed_files)])
                 
    files_1 = [set(sound_names).intersection(tuple(not_sad_f[i])) for i in range(len(not_sad_f))]

    files_2 = set(sound_names).intersection(classes_files.sad_files)
                                                           
    try:                                                      
        if files_1 and (not os.path.exists(files_dir+'/hfc/not sad')):                                                    
            os.mkdir(files_dir+'/hfc/not sad')                                                              
        if files_2 and (not os.path.exists(files_dir+'/hfc/sad')):                                               
            os.mkdir(files_dir+'/hfc/sad')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                     
                                        
    for i in range(len(files_1)):
        for f in files_1[i]:
            try:
                shutil.copy(files_dir+'/hfc/'+(str(f))+'hfc.ogg', files_dir+'/hfc/not sad/'+(str(f))+'hfc.ogg')
            except Exception, e:
                print IOError("No sound file found")
                                                                                         
    for f in files_2:
        try:
            shutil.copy(files_dir+'/hfc/'+(str(f))+'hfc.ogg', files_dir+'/hfc/sad/'+(str(f))+'hfc.ogg')
        except Exception, e:
            print IOError("No sound file found")

    sad_dir = files_dir+'/hfc/sad' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_audio = [scratch_music(i) for i in sad_audio]
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
    not_sad_samples = [scratch_music(i) for i in not_sad_samples]  
    not_sad_x = np.array(not_sad_samples).sum(axis=0) 
    not_sad_X = 0.5*not_sad_x/not_sad_x.max()
    not_sad_Harmonic, not_sad_Percussive = decompose.hpss(librosa.core.stft(not_sad_X))
    not_sad_percussive = istft(not_sad_Percussive)  
    MonoWriter(filename=files_dir+'/hfc/not sad/not_sad_mix_hfc.ogg', format = 'ogg', sampleRate = 44100)(not_sad_percussive) 

def loudness_emotions_remix(files_dir, classes_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param classes_files: neg_and_pos result (read main_svm class documentation)                          
    :param files_dir: data tag dir                                           
    """                                                                                         

    if classes_files.angry_files:
        print (repr(angry_files)+"emotion is angry")

    if classes_files:
        print (repr(classes_files)+"emotion is Not Happy")
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    not_happy_f = list([set(classes_files.sad_files), set(classes_files.angry_files), set(classes_files.relaxed_files)])
                 
    files_1 = [set(sound_names).intersection(tuple(not_happy_f[i])) for i in range(len(not_happy_f))]

    files_2 = set(sound_names).intersection(classes_files.angry_files)
                                                           
    try:                                                      
        if files_1:                                                    
            os.mkdir(files_dir+'/loudness/not happy')                                                              
        if files_2:                                               
            os.mkdir(files_dir+'/loudness/angry')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                      
                                        
    for i in range(len(files_1)):
        for f in files_1[i]:
            try:
                shutil.copy(files_dir+'/loudness/'+(str(f))+'loudness.ogg', files_dir+'/loudness/not happy/'+(str(f))+'loudness.ogg')
            except Exception, e:
                print IOError("No sound file found")
                                                                                         
    for f in files_2:
        try:
            shutil.copy(files_dir+'/loudness/'+(str(f))+'loudness.ogg', files_dir+'/loudness/angry'+(str(f))+'loudness.ogg')
        except Exception, e:
            print IOError("No sound file found")


    sad_dir = files_dir+'/loudness/not happy' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_audio = [scratch_music(i) for i in sad_audio]
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
    angry_audio = [scratch_music(i) for i in angry_audio]
    angry_N = min([len(i) for i in angry_audio])  
    angry_samples = [i[:angry_N]/i.max() for i in angry_audio]  
    angry_x = np.array(angry_samples).sum(axis=0) 
    angry_X = 0.5*angry_x/angry_x.max()
    angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(angry_X))
    angry_harmonic = istft(angry_Harmonic)  
    MonoWriter(filename=files_dir+'/loudness/angry/angry_mix_loudness.ogg', format = 'ogg', sampleRate = 44100)(angry_harmonic) 

def inharmonicity_emotions_remix(files_dir, classes_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param classes_files: neg_and_pos result (read main_svm class documentation)
    :param files_dir: data tag dir                                        
    """                                                                                         
                                                  
    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    not_relaxed_f = list([set(classes_files.sad_files), set(classes_files.angry_files), set(classes_files.happy_files)])
    not_angry_f = list([set(classes_files.sad_files), set(classes_files.relaxed_files), set(classes_files.happy_files)])

    if not_relaxed_f:
        print (repr(not_relaxed_f)+"emotion is not relaxed")

    if not_angry_f:
        print (repr(not_angry_f)+"emotion is not angry")
                 
    files_1 = [set(sound_names).intersection(tuple(not_relaxed_f[i])) for i in range(len(not_relaxed_f))]

    files_2 = [set(sound_names).intersection(tuple(not_angry_f[i])) for i in range(len(not_angry_f))]
                                                           
    try:                                                      
        if files_1:                                                    
            os.mkdir(files_dir+'/inharmonicity/not relaxed')                                                              
        if files_2:                                               
            os.mkdir(files_dir+'/inharmonicity/not angry')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                     
                                        
    for i in range(len(files_1)):
        for f in files_1[i]:
            try:
                shutil.copy(files_dir+'/inharmonicity/'+(str(f))+'inharmonicity.ogg', files_dir+'/inharmonicity/not relaxed'+(str(f))+'inharmonicity.ogg')
            except Exception, e:
                print IOError("No sound file found")
                                                                                         
    for i in range(len(files_2)):
        for f in files_2[i]:
            try:
                shutil.copy(files_dir+'/inharmonicity/'+(str(f))+'inharmonicity.ogg', files_dir+'/inharmonicity/not angry'+(str(f))+'inharmonicity.ogg')
            except Exception, e:
                print IOError("No sound file found")

    sad_dir = files_dir+'/inharmonicity/not relaxed' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_audio = [scratch_music(i) for i in sad_audio]
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
    not_angry_audio = [scratch_music(i) for i in not_angry_audio]
    not_angry_N = min([len(i) for i in not_angry_audio])  
    not_angry_samples = [i[:not_angry_N]/i.max() for i in not_angry_audio]  
    not_angry_x = np.array(not_angry_samples).sum(axis=0) 
    not_angry_X = 0.5*not_angry_x/not_angry_x.max()
    not_angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(not_angry_X))
    not_angry_harmonic = istft(not_angry_Harmonic)  
    MonoWriter(filename=files_dir+'/inharmonicity/not angry/not_angry_mix_inharmonicity.ogg', format = 'ogg', sampleRate = 44100)(not_angry_harmonic)

def contrast_emotions_remix(files_dir, classes_files):
    """                                                                                     
    remix files according to emotion class                                
                                                                                            
    :param classes_files: neg_and_pos result (read main_svm class documentation)
    :param files_dir: data tag dir                                             
    """                                                                                         

    for location in os.walk(files_dir):
             sound_names = [s.split('.')[0] for s in location[2]]   
             break                                                    
                 
    not_relaxed_f = list([set(classes_files.sad_files), set(classes_files.angry_files), set(classes_files.happy_files)])
    not_angry_f = list([set(classes_files.sad_files), set(classes_files.relaxed_files), set(classes_files.happy_files)])

    if not_relaxed_f:
        print (repr(not_relaxed_f)+"emotion is not relaxed")

    if not_angry_f:
        print (repr(not_angry_f)+"emotion is not angry")
                 
    files_1 = [set(sound_names).intersection(tuple(not_relaxed_f[i])) for i in range(len(not_relaxed_f))]

    files_2 = [set(sound_names).intersection(tuple(not_angry_f[i])) for i in range(len(not_angry_f))]
                                                           
    try:                                                      
        if files_1:                                                    
            os.mkdir(files_dir+'/valleys/not relaxed')                                                              
        if files_2:                                               
            os.mkdir(files_dir+'/valleys/not angry')           
    except Exception, e: 
        raise IOError("Sonifications of described sounds haven't been found")                                                                     
                                        
    for i in range(len(files_1)):
        for f in files_1[i]:
            try:
                shutil.copy(files_dir+'/valleys/'+(str(f))+'contrast.ogg', files_dir+'/valleys/not relaxed'+(str(f))+'contrast.ogg')
            except Exception, e:
                print IOError("No sound file found")
                                                                                         
    for i in range(len(files_2)):
        for f in files_2[i]:
            try:
                shutil.copy(files_dir+'/valleys/'+(str(f))+'contrast.ogg', files_dir+'/valleys/not angry'+(str(f))+'contrast.ogg')
            except Exception, e:
                print IOError("No sound file found")

    sad_dir = files_dir+'/valleys/not relaxed' 
    for subdirs, dirs, sounds in os.walk(sad_dir):  
    	sad_audio = [MonoLoader(filename=sad_dir+'/'+sad_f)() for sad_f in sounds]
    sad_audio = [scratch_music(i) for i in sad_audio]
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
    not_angry_audio = [scratch_music(i) for i in not_angry_audio]
    not_angry_N = min([len(i) for i in not_angry_audio])  
    not_angry_samples = [i[:not_angry_N]/i.max() for i in not_angry_audio]  
    not_angry_x = np.array(not_angry_samples).sum(axis=0) 
    not_angry_X = 0.5*not_angry_x/not_angry_x.max()
    not_angry_Harmonic, angry_Percussive = decompose.hpss(librosa.core.stft(not_angry_X))
    not_angry_harmonic = istft(not_angry_Harmonic)  
    MonoWriter(filename=files_dir+'/valleys/not angry/not_angry_mix_contrast.ogg', format = 'ogg', sampleRate = 44100)(not_angry_harmonic) 

#locate all files in data emotions dir
def multitag_emotions_dir(tags_dirs, angry_files, sad_files, relaxed_files, happy_files):
    """                                                                                     
    remix all files according to multitag emotions classes                                

    :param tags_dirs: directories of tags in data                                                                                            
    :param angry_files: files with multitag negative value
    :param sad_files: files with multitag sad value
    :param relaxed_files: files with multitag relaxed value
    :param happy_files: files with multitag happy value
                                                                                                                                                                         
    """                                                                                         
    files_format = ['.mp3', '.ogg', '.undefined', '.wav', '.mid', '.wma', '.amr']

    print ("Sound files evoking anger as emotion: " + repr(angry_files))

    print ("Sound files evoking relaxation as emotion: " + repr(relaxed_files))

    print ("Sound files evoking sadness as emotion: " + repr(sad_files))

    print ("Sound files evoking happiness as emotion: " + repr(happy_files))

    sounds = []
                                                  
    for tag in tags_dirs:
            for types in next(os.walk(tag)):
               for t in types:
                   if os.path.splitext(t)[1] in files_format:
                       sounds.append(t)

    sound_names = []

    for s in sounds:
         sound_names.append(s.split('.')[0])                                                  
                 
    files_0 = set(sound_names).intersection(angry_files)
    files_1 = set(sound_names).intersection(sad_files) 
    files_2 = set(sound_names).intersection(relaxed_files) 
    files_3 = set(sound_names).intersection(happy_files)                                                                     
                                        
    for tag in tags_dirs:
                 for types in next(os.walk(tag)):
                    for t in types:
                        if os.path.splitext(t)[0] in files_0:
                            if t in types:
                                shutil.copy(os.path.join(tag, t), os.path.join(files_dir+ "/emotions/angry/",t))
                        if os.path.splitext(t)[0] in files_1:
                            if t in types:
                                shutil.copy(os.path.join(tag, t), os.path.join(files_dir + "/emotions/sad",t))
                        if os.path.splitext(t)[0] in files_2:
                            if t in types:
                                shutil.copy(os.path.join(tag, t), os.path.join(files_dir + "/emotions/relaxed", t)) 
                        if os.path.splitext(t)[0] in files_3:
                            if t in types:
                                shutil.copy(os.path.join(tag, t), os.path.join(files_dir + "/emotions/happy", t))


from transitions import Machine
import random
import subprocess

#sub-directories examples, you can change the location according to where you might have specified another sounds location
not_happy_dir = ['data/emotions/negative_arousal/sad', 'data/emotions/positive_arousal/angry', 'data/emotions/negative_arousal/relaxed']
not_sad_dir = ['data/emotions/positive_arousal/happy', 'data/emotions/positive_arousal/angry', 'data/emotions/negative_arousal/relaxed']
not_angry_dir = ['data/emotions/positive_arousal/happy', 'data/emotions/negative_arousal/sad', 'data/emotions/negative_arousal/relaxed']
not_relaxed_dir = ['data/emotions/positive_arousal/happy', 'data/emotions/negative_arousal/sad', 'data/emotions/positive_arousal/angry']

def speedx(sound_array, factor):
    """
    Multiplies the sound's speed by a factor
    :param sound_array: your input sound 
    :param factor: your speed up factor                                                                              
    :returns:                                                                                                         
      - faster sound
    """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)]
    return sound_array[np.int32(indices)]

#apply crossfading into scratching method
@memoize
def crossfade(audio1, audio2):
    """ 
    Apply crossfading to 2 audio tracks. The fade function is randomly applied
    :param audio1: your first signal 
    :param audio2: your second signal                                                                            
    :returns:                                                                                                         
      - crossfaded audio
    """
    def quadratic_fade_out(u):  
        return (np.float32([(1 - u[i]) * u[i] for i in range(len(u))]), u*u) 
    def quadratic_fade_in(u):     
        return (np.float32([(1 - u[i]) * u[i] for i in range(len(u))]), np.float32([1 - ((1 - u[i]) * u[i]) for i in range(len(u))]))
    u = audio1/float(len(audio1))            
    if choice([0,1]) == 0:
        amp1, amp2 = quadratic_fade_in(u)      
    else:
        amp1, amp2 = quadratic_fade_out(u)       
    return (audio1 * amp1) + (audio2 * amp2) 

#scratch your records
def scratch(audio):
    """ 
    This function performs scratching effects to sound arrays 
    :param audio: the signal you want to scratch                                                                           
    :returns:                                                                                                         
      - scratched signal
    """
    proportion = len(audio)/16
    def hand_move(audio, rev_audio): #simulation of hand motion in a turntable
        forwards = speedx(audio, randint(2,3))
        backwards = speedx(np.array(rev_audio), randint(2,3))
        forwards = np.append(np.cumsum(forwards).max(), forwards)
        backwards = np.append(np.cumsum(backwards).max(), backwards)
        backwards = np.append(backwards, np.cumsum(backwards).max())
        if len(forwards) > len(backwards):
            cf = crossfade(forwards[len(forwards) - len(forwards[:len(backwards)]):], backwards)
            return np.concatenate([forwards[:len(forwards) - len(forwards[:len(backwards)])], cf])
        else:
            cf = crossfade(forwards, backwards[len(backwards) - len(backwards[:len(forwards)]):])
            return np.concatenate([forwards[:len(forwards) - len(forwards[:len(backwards)])], cf])
    rev_audio = np.array(list(reversed(audio)))
    hand_a = hand_move(audio, rev_audio)
    hand_b = hand_move(audio[:proportion * 12], rev_audio[:proportion*12])
    hand_c = hand_move(audio[proportion:], rev_audio)
    hand_d = hand_move(audio[proportion:proportion*12], rev_audio[:proportion*12])
    hand_e = hand_move(audio, rev_audio[:proportion*2])
    hand_f = hand_move(audio[:proportion*12], rev_audio[proportion*2:proportion*12])
    hand_g = hand_move(audio[:proportion*2], rev_audio[proportion*2:])
    hand_h = hand_move(audio[proportion*3:proportion*12], rev_audio[proportion*3:proportion*12])
    mov = lambda: choice((hand_a, hand_b, hand_c, hand_d, hand_e, hand_f, hand_g, hand_h))
    return np.concatenate([mov(), mov(), mov()])

def scratch_music(audio):
    """ 
    Enclosing function that performs DJ scratching and crossfading to a signal. First of all an arbitrary ammount of times to scratch the audio is chosen to segment the sound n times and also to arbitrarily decide the scratching technique 
    :param audio: the signal you want to scratch                                                                           
    :returns:                                                                                                         
      - scratched recording
    """
    iterations = choice(range(int(Duration()(audio) / 8)))
    for i in range(iterations):
        sound = do_segmentation(audio_input = audio, audio_input_from_filename = False, audio_input_from_array = True, sec_len = choice(range(2,8)), save_file = False) 
        scratches = scratch(sound)
        samples_len = len(scratches)
        position = choice(range(int(Duration()(audio)))) * 44100
        audio = np.concatenate([audio[:position - samples_len], scratches, audio[position + samples_len:]])
    return audio

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
                    x, y = [scratch_music(i) for i in (x,y)]
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
                    	remix_filename = 'data/emotions/remixes/sad/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                    	MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(sad_percussive)
                    	subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def happy_music_remix(self, pos_arous_dir, harmonic = None):
                for subdirs, dirs, sounds in os.walk(pos_arous_dir):  
                    x = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:-1]))()
                    y = MonoLoader(filename=pos_arous_dir+'/'+random.choice(sounds[:]))()
                x, y = [scratch_music(i) for i in (x,y)]
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
		        remix_filename = 'data/emotions/remixes/happy/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
		        MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(happy_percussive)
		        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def relaxed_music_remix(self, neg_arous_dir):
                neg_arousal_h = MusicEmotionStateMachine('remix').sad_music_remix(neg_arous_dir, harmonic = True)
                relaxed_harmonic = istft(neg_arousal_h)
                relaxed_harmonic = 0.5*relaxed_harmonic/relaxed_harmonic.max()
                remix_filename = 'data/emotions/remixes/relaxed/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(relaxed_harmonic)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def angry_music_remix(self, pos_arous_dir):
                pos_arousal_h = MusicEmotionStateMachine('remix').happy_music_remix(pos_arous_dir, harmonic = True)
                angry_harmonic = istft(pos_arousal_h)
                angry_harmonic = 0.5*angry_harmonic/angry_harmonic.max()
                remix_filename = 'data/emotions/remixes/angry/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, format = 'ogg', sampleRate = 44100)(angry_harmonic)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def not_happy_music_remix(self, neg_arous_dir):
                sounds = []
                for i in range(len(neg_arous_dir)):
                    for subdirs, dirs, s in os.walk(neg_arous_dir[i]):                                  
                        sounds.append(subdirs + '/' + random.choice(s))
                x = MonoLoader(filename= random.choice(sounds[:-1]))()
                y = MonoLoader(filename= random.choice(sounds[:]))()
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
                x, y = [scratch_music(i) for i in (x,y)]
                not_happy_x = np.sum((x,y),axis=0) 
                not_happy_X = 0.5*not_happy_x/not_happy_x.max()
                remix_filename = 'data/emotions/remixes/not happy/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename=remix_filename, sampleRate = 44100, format = 'ogg')(not_happy_X)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def not_sad_music_remix(self, pos_arous_dir):
                sounds = []
                for i in range(len(pos_arous_dir)):
                    for subdirs, dirs, s in os.walk(pos_arous_dir[i]):                                  
                        sounds.append(subdirs + '/' + random.choice(s))
                x = MonoLoader(filename= random.choice(sounds[:-1]))()
                y = MonoLoader(filename= random.choice(sounds[:]))()
                x, y = [scratch_music(i) for i in (x,y)]
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
                remix_filename = 'data/emotions/remixes/not sad/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename= remix_filename, sampleRate = 44100, format = 'ogg')(not_sad_X)
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def not_angry_music_remix(self, neg_arous_dir):
                sounds = []
                for i in range(len(neg_arous_dir)):
                    for subdirs, dirs, s in os.walk(neg_arous_dir[i]):                                  
                        sounds.append(subdirs + '/' + random.choice(s))
                x = MonoLoader(filename= random.choice(sounds[:-1]))()
                y = MonoLoader(filename= random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0]
                x, y = [scratch_music(i) for i in (x,y)]
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
                morph = stft.morph(x1 = x,x2 = y,fs = 44100,w1=np.hanning(1025),N1=2048,w2=np.hanning(1025),N2=2048,H1=512,smoothf=0.1,balancef=0.7)
                remix_filename = 'data/emotions/remixes/not angry/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename = remix_filename, sampleRate = 44100, format = 'ogg')(np.float32(morph))
                subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename])
            def not_relaxed_music_remix(self, pos_arous_dir):
                sounds = []
                for i in range(len(pos_arous_dir)):
                    for subdirs, dirs, s in os.walk(pos_arous_dir[i]):                                  
                        sounds.append(subdirs + '/' + random.choice(s))
                x = MonoLoader(filename= random.choice(sounds[:-1]))()
                y = MonoLoader(filename= random.choice(sounds[:]))()
                x_tempo = beat.beat_track(x)[0] 
                y_tempo = beat.beat_track(y)[0]
                x, y = [scratch_music(i) for i in (x,y)] 
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
                morph = stft.morph(x1 = x,x2 = y,fs = 44100,w1=np.hanning(1025),N1=2048,w2=np.hanning(1025),N2=2048,H1=512,smoothf=0.01,balancef=0.7)
                remix_filename = 'data/emotions/remixes/not relaxed/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix.ogg'
                MonoWriter(filename = remix_filename, sampleRate = 44100, format = 'ogg')(np.float32(morph)) 
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
            input_layers = svm_layers()
            input_layers.layer_computation(fscaled, labels)

            fx = input_layers.sum_fx()

            labl = input_layers.best_labels()

            Cs = [1./0.33, 1./0.4, 1./0.6, 1./0.8] #it should work well with less parameter searching
            reg_params = [0.33, 0.4, 0.6, 0.8] 
            kernel_configs = [['linear', 'poly']] #use most significant configurations for music 
            best_estimators = GridSearch(svm_layers(), fx, labl, Cs, reg_params, kernel_configs)
            C, reg_param, kernels_config = best_kernels_output(best_estimators, kernel_configs)
            msvm = main_svm(fx, labl, C[0], reg_param[0], 1./features.shape[1], kernels_config)
            neg_and_pos = msvm.neg_and_pos(files)
            emotions_data_dir()
            multitag_emotions_dir(tags_dirs, neg_and_pos.angry_files, neg_and_pos.sad_files, neg_and_pos.relaxed_files, neg_and_pos.happy_files)
        if sys.argv[2] in ('None'):
            files = descriptors_and_keys(files_dir, None)._files
            features = descriptors_and_keys(files_dir, None)._features
            fscaled = feature_scaling(features)
            labels = KMeans_clusters(fscaled)
            input_layers = svm_layers()
            input_layers.layer_computation(fscaled, labels)

            fx = input_layers.sum_fx()

            labl = input_layers.best_labels()

            Cs = [1./0.33, 1./0.4, 1./0.6, 1./0.8] #it should work well with less parameter searching
            reg_params = [0.33, 0.4, 0.6, 0.8] 
            kernel_configs = [['linear', 'poly']] #use most significant configurations for music 
            best_estimators = GridSearch(svm_layers(), fx, labl, Cs, reg_params, kernel_configs)
            C, reg_param, kernels_config = best_kernels_output(best_estimators, kernel_configs)
            msvm = main_svm(fx, labl, C[0], reg_param[0], 1./features.shape[1], kernels_config)
            neg_and_pos = msvm.neg_and_pos(files)

            #only remix the better extracted emotions that can be obtained from descriptions
            bpm_emotions_remix(files_dir, neg_and_pos.sad_files, neg_and_pos.happy_files)
            attack_emotions_remix(files_dir, neg_and_pos.sad_files, neg_and_pos.angry_files)
            dissonance_emotions_remix(files_dir, neg_and_pos.relaxed_files, neg_and_pos.angry_files)
            mfcc_emotions_remix(files_dir, neg_and_pos.relaxed_files, neg_and_pos.angry_files)
            centroid_emotions_remix(files_dir, neg_and_pos.relaxed_files, neg_and_pos.angry_files)
            hfc_emotions_remix(files_dir, neg_and_pos)
            loudness_emotions_remix(files_dir, neg_and_pos)
            inharmonicity_emotions_remix(files_dir, neg_and_pos)
            contrast_emotions_remix(files_dir, neg_and_pos)
        if sys.argv[2] in ('False'): 
		me = MusicEmotionStateMachine("Johnny") #calling Johnny                        
		me.machine.add_ordered_transitions()    #Johnny is very sensitive                  
		                                                          
		while(1):                                                 
		    me.next_state()                                       
		    if me.state == random.choice(me.states):              
		        if me.state == 'happy':                           
		            print me.state                                
		            me.happy_music_remix('data/emotions/positive_arousal/happy', harmonic = None)
		        if me.state == 'sad':                  
		            print me.state                     
		            me.sad_music_remix('data/emotions/negative_arousal/sad', harmonic = None)  
		        if me.state == 'angry':                
		            print me.state                     
		            me.angry_music_remix('data/emotions/positive_arousal/angry')
		        if me.state == 'relaxed':              
		            print me.state                     
		            me.relaxed_music_remix('data/emotions/negative_arousal/relaxed')
		        if me.state == 'not happy':              
		            print me.state                     
		            me.not_happy_music_remix(not_happy_dir)
		        if me.state == 'not sad':              
		            print me.state                     
		            me.not_sad_music_remix(not_sad_dir)
		        if me.state == 'not angry':              
		            print me.state                     
		            me.not_angry_music_remix(not_angry_dir)
		        if me.state == 'not relaxed':              
		            print me.state                     
		            me.not_relaxed_music_remix(not_relaxed_dir)
                                                       
    except Exception, e:                     
        logger.exception(e)
