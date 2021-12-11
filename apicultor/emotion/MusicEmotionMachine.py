#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils.dj import *
from ..machine_learning.cache import *
from ..machine_learning.cross_validation import *
from ..machine_learning.fairness import *
from ..machine_learning.dependency import *
from ..machine_learning.explain import *
from ..gradients.descent import SGD
from ..constraints.bounds import dsvm_low_a as la
from ..constraints.bounds import dsvm_high_a as ha
from ..constraints.bounds import es
from ..constraints.tempo import same_time
from ..constraints.time_freq import *
from ..gradients.subproblem import *
from ..utils.data import get_files, get_dics, desc_pair, read_file, read_attention_file, read_good_labels
from ..utils.algorithms import morph
import time
from collections import Counter, defaultdict
from itertools import combinations, permutations
import numpy as np                                                      
import matplotlib.pyplot as plt                                   
import os, sys 
import json                                                          
from ..sonification.Sonification import write_file, hfc_onsets
from soundfile import read  
from sklearn.preprocessing import StandardScaler as sc
from sklearn.preprocessing import LabelBinarizer as lab_bin
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import MinMaxScaler
#pca is a dimensionality reduction method that takes x values and maps these to a reduced form that minimizes
#the squared error by looking for the smallest distance, so it gets a narrower version of the dataset
from sklearn.decomposition import PCA as pca
from sklearn.cluster import KMeans
from random import *
from sklearn.utils.extmath import safe_sparse_dot as ssd
from librosa.core import stft
import pandas                                   
import shutil
import logging
import warnings

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#emotion classification
class descriptors_and_keys():
    """
    get all descriptions given descriptor keys
    :param files_dir: data tag dir if not performing multitag classification. data dir if performing multitag classification
    :param multitag: if True, will classify all downloaded files and remix when performing emotional state transition 
    """                   
    def __init__(self, tags_dir,  multitag):
        self.multitag = None
        self._files_dir = tags_dir
        if multitag == None or multitag == False:                                                
            raise IndexError("Need lots of data for emotion classification")                                    
        elif multitag == True:                                              
            self._files = np.hstack([get_files(tag) for tag in self._files_dir])        
            self._dics = [[] for tag in self._files_dir]
            try:                             
                for tag in range(len(self._files_dir)):                       
                    for subdir, dirs, files in os.walk(self._files_dir[tag]+'/descriptores'):                                               
                        for f in files:                               
                            with open(subdir + '/' + f) as read:                                                              
                                self._dics[tag].append(json.load(read)) 
            except:                                              
                if not os.path.exists(self._files_dir[tag]+'/descriptores'):
                    print ("No readable MIR data found")
            self._dics = np.hstack(self._dics)          
        for i,x in enumerate(np.vstack((self._files, [len(i) for i in self._dics])).T): 
            if x[1] != str(13): #if dictionary of a file is shorter...     
                fname = i                                                  
                for i in self._files:                                            
                    fname = i[fname] #...finds the name of the .json file  
                for tag in tags_dirs:                                      
                    for subdir, dirs, sounds in os.walk(tag+'/descriptores'):
                        for s in sounds:                                   
                            if fname == s:                                 
                                print((os.path.abspath(tag+'/descriptores/'+s) +" has less descriptors and will be discarded for clustering and classification. This happens because there was an error in MIR analysis, so the sound has to be analyzed again after being processed")) 
                                os.remove((os.path.abspath(tag+'/descriptores/'+s))) #... and deletes the less suitable .json file
        self._files = list(self._files)
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
        dp = desc_pair(self._files,self._dics)
        self._features = dp._features
        self._files = dp._files

def feature_scaling(f):
    """
    scale features
    :param features: combinations of features                                                                               
    :returns:                                                                                                         
      - normalized column features
    """    
    F = f.copy()
    for i in range(len(F.T)):
        F[:,i] = F[:,i] / F[:,i].max()
    return F  

def zero_mean_unit_var(x,val):
    """
    scale features with zero mean and unit variance
    :param x: dataset                                                                              
    :param val: value or dataset to normalize
    :returns:                                                                                                         
      - scaled column features with mean and standard deviation
    """ 
    x = val - x.mean()
    return x / np.std(x)

def KMeans_clusters(fscaled):
    """
    KMeans clustering for features                                                           
    :param fscaled: scaled features                                                                                         
    :returns:                                                                                                         
      - labels: classes         
    """
    labels = (KMeans(init = pca(n_components = 4).fit(fscaled).components_, n_clusters=4, n_init=1, precompute_distances = True, random_state = 0, n_jobs = -1).fit(fscaled).labels_)
    return labels

def pad_features(x,hyp,column,sensible_column,condition):
    """
    A function that takes a hypothesis condition to bulge only
    values that satisfy the given condition. This is a construction function
    that can be useful for noisy datasets
    :param x: dataset                                                                              
    :param hypothesis: threshold value
    :param column: feature that proves hypothesis
    :param sensible_column: index of feature to pad while related to the conditional features 
    :param condition: bool for logical condition 
    :returns:                                                                                                         
      - Dataset with padded features  
    """
    rows = []
    for l in range(x.shape[1]):
        if l != sensible_column:
            rows.append(x[:,l])
        else:
            r = []
            for row in range(len(x)):
                if condition is True:
                    if x[:,column][row] > hyp:
                        r.append(0)
                    else:
                        r.append(x[:,sensible_column][row])  
                else:        
                    if x[:,column][row] < hyp:
                        r.append(x[:,sensible_column][row])
                    else:
                        r.append(0)
            rows.append(np.array(r))
    return np.vstack(rows).T

def assign(x,val,index,axis):
    """
    A function that assigns the specified array values
    in an index if it is a row or to a column if it is a column index
    by vertically stacking the array columns with the values to assign
    :param x: array to assign values                                                                              
    :param val: if int type is given, an array of the shape or size of the
    matrix is going to be given to assign the value, else the given array must
    be of the same size as the size in the axis of the array
    :param index: index to assign values
    :param axis: row index (if 0) or column indx (if greater)
    :returns:                                                                                                         
      - stack.T: the input array with assigned values    
    """
    if index == 0:
        if type(val) == int:
            if axis == 0:
                stack = [val for i in range(x.shape[0])]
            else:
                stack = [val for i in range(x.shape[1])]
        else:
            if axis == 0:
                if len(val) != x.shape[0]:
                    raise IndexError('Values size differ from the original row size')
            else:    
                if len(val) != x.shape[1]:
                    raise IndexError('Values size differ from the original column size')
            stack = val
    else:    
        if axis == 0:
            stack = x[0]
        else:
            stack = x[:,index]
        for col in range(x.shape[1]):
            if col != index:
                if col+1 != index:
                    try:
                        stack = np.vstack((stack,x[:,col+1]))
                    except Exception as e:
                        stack = np.vstack((stack,x[:,col]))
                else:
                    stack = np.vstack((stack,val))
    return stack.T

class deep_support_vector_machines(object):
    """
    Functions for Deep Support Vector Machines                                               
    :param features: scaled features                                                                                        
    :param labels: classes                                                                                            
    """ 
    def __init__(self, kernel1, kernel2):
        self.kernel1 = None
        self.kernel2 = None
        self.arch = defaultdict(list)
    def apply_duals(self,layer):
        """
        A function that updates the model in class with a given layer's variables                                    
        :returns:                                         
          - The current model with updated coefficients       
        """
        if isinstance(layer.bias, self): #coefficients are known before bias                
            #if there is a retrain, optimization of alpha is based on applied dual 
            self.dual_coefficients = layer.dual_coefficients 
            self.sv_locs = layer.sv_locs 
            #tell the object it is using previous properties
            self.written = True
        else:                 
            raise IndexError('ERROR: Trying to update with an empty layer')
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
        #print(x.shape)
        #print(y.shape)

        pk = x @ y.T

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
        return x @ y
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

        sk = x @ y.T

        sk *= gamma

        sk += c

        np.tanh(np.array(sk, dtype='float64'), np.array(sk, dtype = 'float64'))
        return np.array(sk, dtype = 'float64')
    @classmethod
    @memoize
    def rbf_kernel(self, x, y, gamma):
        """
        Custom radial basis function, similarities of vectors and as a surrogate modelling predictor using 
        a radial basis function kernel
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
        #if std === mean all values decay the same
        np.exp(rbfk,rbfk)
        return np.array(rbfk)

    def fit_model(self, features,labels, kernel1, kernel2, C, reg_param, gamma, learning_rate):
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
        self.learning_rate = learning_rate

        features = np.ascontiguousarray(features)

        multiclass = False

        if not 0 in labels:                                      
            raise IndexError("Targets must start from 0")

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
            self.instances = list(combinations(list(range(self.n_class)), 2))
            self.classes = np.unique(self.targets)
            self.classifications = len(self.instances)
            self.classesm = defaultdict(list) 
            self.trained_features = defaultdict(list) 
            self.indices_features = defaultdict(list) 
            self.a = defaultdict(list)
            self.svs = defaultdict(list)
            self.ns = defaultdict(list)
            self.nvs = defaultdict(list)
            self.dual_coefficients = [[] for i in range(self.n_class - 1)]
        #to pipeline
        for c in range(self.classifications):   

            self.classesm[c] = np.concatenate((self.targets[self.targets==self.instances[c][0]],self.targets[self.targets==self.instances[c][1]]))
            self.indices_features[c] = np.concatenate((np.where(self.targets==self.instances[c][0]),np.where(self.targets==self.instances[c][1])), axis = 1)[0]
            self.trained_features[c] = np.ascontiguousarray(np.concatenate((features[self.targets==self.instances[c][0]],features[self.targets==self.instances[c][1]])))

            self.classesm[c] = lab_bin(0,1).fit_transform(self.classesm[c]).T[0]

            n_f = len(self.trained_features[c])
            n_f0 = len(self.classesm[c][self.classesm[c]==0])
                                                                     
            lab = self.classesm[c] * 2 - 1
                                                                   
            if self.kernel2 == "linear":                                                                      
                kernel = self.linear_kernel_matrix
                self.Q = kernel(self.trained_features[c], self.trained_features[c].T) * lab            
            if self.kernel2 == "poly":                                
                kernel = self.polynomial_kernel
                self.Q = kernel(self.trained_features[c], self.trained_features[c], self.gamma) * lab        
            if self.kernel2 == "rbf":                                                                      
                kernel = self.rbf_kernel
                self.Q = kernel(self.trained_features[c], self.trained_features[c], self.gamma) * lab              
            if self.kernel2 == "sigmoid":                             
                kernel = self.sigmoid_kernel 
                self.Q = kernel(self.trained_features[c], self.trained_features[c], self.gamma) * lab          
            if self.kernel2 == None:                                  
                print ("Apply a kernel")
                                                                      
            class_weight = float(n_f)/(2 * np.bincount(self.classesm[c]))

            a = np.zeros(n_f)

            p1 = np.argsort([self.Q[i,i] for i in range(n_f0-1)])
            p2 = np.argsort([self.Q[i,i] for i in range(n_f0, n_f)])+n_f0  

            iterations = 0                                            
            diff = 1. + reg_param 
            for i in range(20):
                a0 = a.copy()
                for j in range(min(p1.size, p2.size)): 
                    g1 = g(lab[p1[j]], a, self.Q[p1[j]])
                    g2 = g(lab[p2[j]], a, self.Q[p2[j]])
                    w0 = es(a, lab, self.trained_features[c]) 
                    direction_grad = self.learning_rate * ((w0 * g2) + ((1 - w0) * g1))

                    if (direction_grad * (g1 + g2) <= 0.):        
                        break

                    a[p1[j]] += direction_grad #do step
                    a[p2[j]] += direction_grad #do step
                #active learning
                a = la(a) #satisfy constraints for lower As
                a = ha(a, class_weight[self.classesm[c]], self.C) #satisfy constraints for higher As using class weights
                diff = Q_a(a, self.Q) - Q_a(a0, self.Q)
                iterations += 1
                if (diff < reg_param):
                    break

            #alphas are automatically signed and those with no incidence in the hyperplane are 0 complex or -0 complex, so alphas != 0 are taken 
            a = SGD(a, lab, self.Q * lab, self.learning_rate)
            a = ha(a, class_weight[self.classesm[c]], self.C) #keep up with the penalty boundary
            a = ha(a * -1, class_weight[self.classesm[c]], self.C)
            a *= -1
                              
            self.ns[c].append([])
            self.ns[c].append([])
            #active learning    
            for dc in range(len(a)):
                if a[dc] != 0.0:
                    if a[dc] > 0.0:
                        self.ns[c][1].append(self.indices_features[c][dc])
                    else:
                        self.ns[c][0].append(self.indices_features[c][dc])
                else:
                    pass
                                                                      
            self.a[c] = a

        ns = defaultdict(list)
        for i, (j,m) in enumerate(self.instances):
            ns[j].append(self.ns[i][0])
            ns[m].append(self.ns[i][1])

        self.ns = [np.unique(np.concatenate(ns[i])) for i in range(len(ns))]

        self.dual_coefficients = [[] for i in range(self.n_class)]
        #split the dataset indexes according to category to stop omission biases
        for i, (j, m) in enumerate(self.instances):
            xnj = np.where(self.targets==self.instances[i][0])
            #this sum is reduced at the next instance
            xnm = np.where(self.targets==self.instances[i][1])
            aj = np.where(np.isin(xnj, self.ns[j]))[1]
            am = np.where(np.isin(xnm, self.ns[m]))[1]
            self.dual_coefficients[j].append(self.a[i][aj])
            self.dual_coefficients[m].append(self.a[i][am])
        try:
            self.dual_coefficients = np.hstack([np.vstack(self.dual_coefficients[i]) for i in range(len(self.dual_coefficients))])
        except Exception as e:
            raise ValueError('pipelineError: cache is hitting previous data')
        self.nvs = [len(i) for i in self.ns]
        
        self.ns = np.concatenate(self.ns).astype(np.uint8)

        self.svs = features[self.ns] #update support vectors

        self.sv_locs = np.cumsum(np.hstack([[0], self.nvs]))

        self.w = [] #update weights given common support vectors, the other values helped making sure it wasn't restarting

        for class1 in range(self.n_class):
            # SVs for class1:
            sv1 = self.svs[self.sv_locs[class1]:self.sv_locs[class1 + 1], :]
            ys1 = self.classesm[c][self.sv_locs[class1]:self.sv_locs[class1 + 1]]
            for class2 in range(class1 + 1, self.n_class):
                # SVs for class1:
                sv2 = self.svs[self.sv_locs[class2]:self.sv_locs[class2 + 1], :]
                ys2 = self.classesm[c][self.sv_locs[class2]:self.sv_locs[class2 + 1]]
                # dual coef for class1 SVs:
                alpha1 = self.dual_coefficients[class2 - 1, self.sv_locs[class1]:self.sv_locs[class1 + 1]]
                # dual coef for class2 SVs:
                alpha2 = self.dual_coefficients[class1, self.sv_locs[class2]:self.sv_locs[class2 + 1]]
                try:  
                    # build weight for class1 vs class2
                    fair_ij = ind_fairness(sv1,ssd(alpha1, sv1),ys1)
                    print("Conditional parity at class ",class1," targeting is ",str(fair_ij)) 
                    fair_group = fair_ij ** 2
                    self.w.append(ssd(alpha1, sv1)+ ssd(alpha2, sv2)) 
                    print("Group fairness at class ",class1," targeting is ",str(fair_group)) 
                    fair_ij = ind_fairness(sv2,ssd(alpha2, sv2),ys2) 
                    print("Conditional parity at class ",class2," targeting is ",str(fair_ij)) 
                    fair_group = fair_ij ** 2 
                    print("Group fairness at class ",class2," targeting is ",str(fair_group)) 
                    old_loss = np.mean((ys1-(ssd(alpha1, sv1)*sv1).T)**2)  
                    unprot1 = unprotection_score(old_loss,(ssd(alpha1, sv1)*sv1).T,ys1) 
                    print("Conditional procedure accuracy equality targeting ",class1," is ",str(unprot1)) 
                    old_loss = np.mean((ys2-(ssd(alpha2, sv2)*sv2).T)**2)  
                    unprot2 = unprotection_score(old_loss,(ssd(alpha2, sv2)*sv2).T,ys2) 
                    print("Conditional procedure accuracy equality targeting ",class2," is ",str(unprot2)) 
                except Exception as e: 
                    raise ValueError("Can't compute individual fairness with empty support values")
        if self.kernel1 == "poly":
            kernel1 = self.polynomial_kernel
        if self.kernel1 == "sigmoid":
            kernel1 = self.sigmoid_kernel
        if self.kernel1 == "rbf":
            kernel1 = self.rbf_kernel 
        if self.kernel1 == "linear":
            kernel1 = self.linear_kernel_matrix

        try:          
            self.bias = []                
            for class1 in range(self.n_class):                              
                sv1 = self.svs[self.sv_locs[class1]:self.sv_locs[class1 + 1], :]
                for class2 in range(class1 + 1, self.n_class):                  
                    sv2 = self.svs[self.sv_locs[class2]:self.sv_locs[class2 + 1], :]
                    if kernel1 == self.linear_kernel_matrix:                                                            
                        self.bias.append(-((kernel1(sv1, self.w[class1].T).max() + kernel1(sv2, self.w[class2].T).min())/2))
                    else:                                                                                                     
                        self.bias.append(-((kernel1(sv1, self.w[class1], self.gamma).max() + kernel1(sv2, self.w[class2], self.gamma).min())/2))
        except Exception as e:          
            print("Can't find a bias")  
        return self 

    def decision_function(self, features):
        """
        Compute the distances from the separating hyperplane (dimensionality reduction)           
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
            self.k = self.svs @ features.T
        else:           
            self.k = kernel(self.svs, features, self.gamma) #svs, sv_locs, dual_coefficients, n_class, bias    

        start = self.sv_locs[:self.n_class]
        end = self.sv_locs[1:self.n_class+1]
        c = [ sum(self.dual_coefficients[ i ][p] * self.k[p] for p in range(start[j], end[j])) +
              sum(self.dual_coefficients[j-1][p] * self.k[p] for p in range(start[i], end[i]))
                for i in range(self.n_class) for j in range(i+1,self.n_class)]

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
        self.decision = self.decision_function(features)
        #proba threshold
        self.proba = sigmoid(self.decision)
        return np.argmax(self.proba,axis=1)

#classify all the features using different kernels (different products) 
class svm_layers(deep_support_vector_machines):
    """
    Functions for Deep Support Vector Machines layers                                        
    :param features: scaled features                                                                                        
    :param labels: classes                                                                                            
    """ 
    def __init__(self):
        super(deep_support_vector_machines, self).__init__()
    def layer_computation(self, features, labels, features_test, labels_test, kernel_configs,criteria,intersections,logic,track_conflicts=None,fig_name=None):
        """
        Computes vector outputs and labels                              
        :param features: scaled features                                
        :param labels: classes                
        :param features_test: features in model testing                               
        :param labels_test: classes in model testing     
        :param kernel_configs: list of tuples of kernels for layers and hidden layers respectively                             
        :param criteria: explanation criteria. This argument takes 'mean','var','std' or a number
        :param intersections: indexes of relevant features for explanation
        :param logic (type(logic) == list): column-wise explanation logic of bool values
        :param track_conflicts (type(track_conflicts) == list or type(track_conflicts) == bool): if None, misclassified data indexes are stored in a list. If a list is given, misclassified values are updated in the list
        :param fig_name (type(fig_name) == str or type(fig_name) == bool): if filepath, plots of layer explanations will be stored at desired location         
        :returns:                                         
          - self: layers training instance              
        """
        #classes are predicted even if we only use the decision functions as part of the output data to get a better scope of classifiers
        self.fxs = []
        self.targets_outputs = []
        self.scores = []
        self.best_layer = []
        features_test = np.float64(features_test)
        sample_weight = float(len(features)) / (len(np.array(list(set(labels)))) * np.bincount(labels))
        #C = 1/.689 for mean and median based on outliers boundary of noise data
        #C = 1/.56 for robust mean and median based on observable anomalies in noise data
        #C = 1/.269 for outliers boundary based on std in noise data       
        Cs = [1./0.2, 1/.269, 1./0.3, 1./0.4, 1./0.5, 1/.56, 1./0.6, 1/.689, 1./0.8]
        reg_params = [0.2, .269, 0.3, 0.4, 0.5, .56, 0.6, .689, 0.8]
        self.kernel_configs = kernel_configs
        data_conflicts = None
        #to pipeline
        #namespace dtypes
        for i in range(len(self.kernel_configs)):
            print ("******LAYER ",i,"******")
            if i > 0:
                prev_conflicts = grid_conflicts
            best_estimator, best_model, grid_conflicts = GridSearch(self, features, labels, Cs, reg_params, [self.kernel_configs[i]],criteria,intersections,logic,track_conflicts)
            print((("Best estimators are: ") + str(best_estimator['C'][0]) + " for C and " + str(best_estimator['reg_param'][0]) + " for regularization parameter"))
            print ("Sanity checking best model")
            try:
                pred_c = best_model.predictions(np.array(features_test), np.array(labels_test))
            except Exception as e:
                print('pipelineError: cache is hitting previous data')
                continue
            err = score(labels_test[:pred_c.size], pred_c)
            #store scores and weights in class
            #check if demographic parity exists
            fairness = p_rule(pred_c,labels_test, best_model.w,features_test,best_model.proba)    
            print("Statistical parity is: ",fairness)                     
            #explain
            if fig_name == None:
                fig = None
            else:
                fig = fig_name+'_'+str(i)+'.png'
            parent_explanations, child_explanations, visuals = explain(best_model,features_test,labels_test,criteria,intersections,logic,fig=fig)
            print('Parent explanation:',parent_explanations)
            print('Child explanation:',child_explanations)
            #compute BTC
            btc = BTC(labels_test, pred_c)   
            for tc in range(len(btc)):
                print('Future backward trust compatibility for target',tc,'is',btc[tc]) 
            #compute BEC
            bec, cons = BEC(labels_test, pred_c)   
            for ec in range(len(bec)):
                print('Future backward error compatibility for target',ec,'is',bec[ec])  
            if 0 < i <= 1:
                self.sample_conflicts = []
            if 0 < i:    
                for con in range(len(grid_conflicts)):
                    con_now=grid_conflicts[con][0]
                    try:
                        con_prev=prev_conflicts[con][0]
                    except Exception as e: 
                        print('Previous conflict data missed during learning error')
                        con_prev = []
                    where_conflicts=np.isin(con_now,con_prev)
                    self.sample_conflicts.append(con_now[where_conflicts])
                print('Done remembering')        
            #min cost trade-off    
            self.scores.append(err)
            self.fxs.append(best_model.decision)            
            self.targets_outputs.append(best_model.predictions(np.float64(features), labels))
            self.best_layer.append(best_model)
        if len(self.fxs) <= 1:
            raise IndexError("Your models need retraining after giving bad information. Most likely data that doesn't scale very good enough")

        return self

    def store_array(self, array, output_dir):
        """
        Save values in a .csv file for further usage
        """  
        array = pandas.DataFrame(array)
        array.to_csv(output_dir + '/attention.csv')  

    def attention(self,features,targets):
        """
        A function that averages feasibility in the model                         
        """
        return attention(features,targets,self.w)

    def best_labels(self):
        """
        Get the labels with highest output in the input layers instances and the weights with min errors      
        """ 
        self.targets_outputs = np.int32([Counter(np.int32(self.targets_outputs)[:,i]).most_common()[0][0] for i in range(np.array(self.targets_outputs).shape[1])])
        self.best_layer = self.best_layer[np.argmin(self.scores)]
        return self.targets_outputs

    def store_good_labels(self, files, output_dir):
        """
        Store best processed targets during learning process
        """  
        d = defaultdict(list) 
        for i in range(len(files)):
            d[files[i]] = self.targets_outputs[i] 
        array = pandas.DataFrame(d, index = list(range(1)))
        array.to_csv(output_dir + '/attention_labels.csv')  

    def store_array(self,fname, arr):  
        """
        Save data in a specified directory
        """
        with open(fname,'wb') as file:
            np.save(file, arr)  

    def get_npy(self,fname):  
        """
        Get npy data in a specified directory
        """
        with open(fname,'rb') as file:
            return np.load(file)                   
                  
    def save_checkpoint(self,fdir,signature):  
        """
        Save model data in a specified directory
        """
        with open(fdir+'models/'+signature+'_modelsvs.npy','wb') as file:
            np.save(file, self.best_layer.svs)
        with open(fdir+'models/'+signature+'_modeltargets.npy','wb') as file:
            np.save(file, self.best_layer.targets_outputs)  
        with open(fdir+'models/'+signature+'_modelsvs_locs.npy','wb') as file:
            np.save(file, self.best_layer.sv_locs)      
        with open(fdir+'models/'+signature+'_modeldual_coef.npy','wb') as file:
            np.save(file, self.best_layer.dual_coefficients) 
        with open(fdir+'models/'+signature+'_modelnclasses.npy','wb') as file:
            np.save(file, self.best_layer.n_class) 
        with open(fdir+'models/'+signature+'_modelbias.npy','wb') as file:
            np.save(file, self.best_layer.bias) 
        #with open(fdir+'models/'+signature+'_modelwslope.npy','wb') as file:
        #    np.save(file, self.best_layer.w) 

    def save_buffer_checkpoint(self,signature=None):  
        """
        Save model data in buffers
        """
        if signature == None:
            signature = ''
        else:
            pass
        try:
            self.arch
        except Exception as e:
            self.arch = defaultdict(list)
        self.arch[signature+'support_vectors'] = self.svs
        #self.arch[signature+'yhat'] = self.targets_outputs 
        self.arch[signature+'support_vector_locations'] = self.sv_locs     
        self.arch[signature+'dual_coefficients'] = self.dual_coefficients 
        self.arch[signature+'ysize'] = self.n_class
        self.arch[signature+'bias'] = self.bias     
        #self.arch[signature+'w'] = self.w    

    def read_checkpoint(self,fdir,signature):  
        """
        Read model data in a specified directory
        """
        with open(fdir+'models/'+signature+'_modelsvs.npy','rb') as file:
            self.svs=np.load(file)
        with open(fdir+'models/'+signature+'_modeltargets.npy','rb') as file:
            self.best_labels=np.load(file)  
        with open(fdir+'models/'+signature+'_modelsvs_locs.npy','rb') as file:
            self.sv_locs=np.load(file)      
        with open(fdir+'models/'+signature+'_modeldual_coef.npy','rb') as file:
            self.dual_coefficients=np.load(file)
        with open(fdir+'models/'+signature+'_modelnclasses.npy','rb') as file:
            self.n_class=np.load(file)
        with open(fdir+'models/'+signature+'_modelbias.npy','rb') as file:
            self.bias=np.load(file)             
        #with open(fdir+'models/'+signature+'_modelwslope.npy','rb') as file:
        #    self.w=np.load(file) 

    def reduce_label_noise(self,x,y,hyp,column,condition,target):
        """
        Use it to stop your model from decaying. Given an explanation, this method
        relabels targeted data with the explained relations such that all misclassified
        data is assigned dependent values. It is a massage method to eliminate label noise
        from a model decay after classification.
        :param x: dataset                               
        :param y: classes                                            
        :param hyp: a string between 'mean', 'var' or 'std' or a numerical value
        :param column: target of hypothesis                           
        :param condition: a bool expressing condition of hypothesis
        :param target: decayed target 
        :returns:                                         
            - y: noiseless targets
        """
        if hyp == 'mean':
            limit = x[:,column].mean()
        elif hyp == 'var':
            limit = np.var(x[:,column])
        elif hyp == 'std':
            limit = np.std(x[:,column])
        else:
            limit = hyp
        for i in np.unique(y):
            if condition is True:
                y[x[:,column]>limit] = target
            else:
                y[np.logical_not(x[:,column]>limit)] = target        
        return y                
            
    def compile_features(self,x,y,conflicts,hyp,column,condition):
        """
        Given backward error points and a dependency explanation, this method
        restores dependencies with the explained relations such that all misclassified
        data is assigned their dependent values and removes its conflicts from dataset
        and targets. It is a massage method to privilege a dependent dataset
        :param x: dataset                               
        :param y: classes                
        :param conflicts: an array of backward error indexes                              
        :param hyp: a string between 'mean', 'var' or 'std' or a numerical value
        :param column: perturbed feature of dependency                            
        :param condition: a bool expressing condition of hypothesis
        :returns:                                         
            - x: compiled features
            - y: compiled targets
        """
        nerrors = []
        for i in range(len(conflicts)):
            nerrors.append(len(conflicts[i]))
        dependency = np.argmin(nerrors)  
        if hyp == 'mean':
            limit = x[:,column].mean()
        elif hyp == 'var':
            limit = np.var(x[:,column])
        elif hyp == 'std':
            limit = np.std(x[:,column])
        else:
            limit = hyp
        for i in range(len(conflicts)):
            if i != dependency:
                if condition is True:
                    y[conflicts[i]][x[conflicts[i]][:,column]>limit] = dependency
                else:
                    y[conflicts[i]][np.logical_not(x[conflicts[i]][:,column]>limit)] = dependency        
        return np.delete(x,conflicts[dependency],axis=0), np.delete(y,conflicts[dependency])            

    def supress(x,y,conflicts):
        """
        Create a new dataset by deleting its conflicts
        """
        conflicts_idx = np.unique(np.sort(np.hstack(conflicts)))
        for i in conflicts_idx:
            x[i] = False
            y[i] = -1         
        return x[y!=-1]    

    def get_conflicts(self,x,found,cols):
        """
        Find learning conflicts by creating a new prediction instance with
        a perturbed input that preserves their relevant features                        
        :param x: input dataset  
        :param found: targets in production
        :param cols (type(cols) == list): list of columns equal to the number.
        of targets that are not relevant for prediction. Eg.: [0,3,3]
        :returns:                                         
            - conflicts: indexes of values with conflicts
        """
        size = 0
        resamples = 0
        for yi in range(self.n_class):
            xshifted = np.copy(x)
            resampled_arr = sample(list(xshifted[found==yi][:,cols[yi]]),xshifted[found==yi][:,cols[yi]].size)
            if cols[yi] == 0:
                stack = resampled_arr
            else:    
                stack = xshifted[found==yi][:,0]
            for col in range(xshifted.shape[1]):
                if col != cols[yi]:
                    if col+1 != cols[yi]:
                        try:
                            stack = np.vstack((stack,xshifted[found==yi][:,col+1]))
                        except Exception as e:
                            stack = np.vstack((stack,xshifted[found==yi][:,col]))
                    else:
                        stack = np.vstack((stack,resampled_arr))
                        resamples += 1
            xshifted = stack.T
            try:
                predicted = self.predictions(xshifted,None)
                #compute BTC
                btc = BTC(found, predicted)   
                #print('BTC for target',size,'if changed is',btc,'at step',yi)   
                #compute BEC
                if yi > 0:
                    try: #pythonic
                        prev_cons = cons
                    except Exception as e:
                        pass
                bec, cons = BEC(found, predicted)   
                #print('BEC for target',size,'if changed is',bec,'at step',yi)  
                if 0 == yi:
                    conflicts = []
                elif 1 == resamples:
                    conflicts = []
                size += 1
                for con in range(len(cons)):
                    con_now=cons[con][0]
                    try:
                        con_prev=cons[con][0]
                    except Exception as e: 
                        con_prev = []
                    where_conflicts=np.isin(con_now,con_prev)
                    conflicts.append(con_now[where_conflicts])
            except Exception as e:
                #print('Skipping backward check from unfound targets')
                pass
        try:        
            return conflicts
        except Exception as e:
            return []
        

    def compa_grade(self,x,found,cols,hyp,depcol,condition):
        """
        Keep compatibility after a prediction task using positions of conflicts
        to resolve dataset dependencies
        :param x: input dataset  
        :param found: targets in production
        :param cols (type(cols) == list): list of columns equal to the number.
        of targets that are not relevant for prediction. Eg.: [0,3,3]
        :param hyp: a string between 'mean', 'var' or 'std' or a numerical value
        :param depcol: perturbed feature of dependency                            
        :param condition: a bool expressing condition of hypothesis
        :returns:                                         
            - xresolved: usable dataset
        """
        conflicts = self.get_conflicts(x,found,cols)
        if conflicts != []:
            xresolved, ysolved = self.compile_features(x,found,conflicts,hyp,depcol,condition)
            return xresolved 
        else:
            return x #model decay 
            
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
    def __init__(self, S, lab, C, reg_param, gamma, kernels_config, output_dir):
        super(deep_support_vector_machines, self).__init__()
        self._S = S
        self.output_dir = output_dir
        self.fit_model(self._S, lab, kernels_config[0], kernels_config[1], C, reg_param, gamma, 0.8)
        self._labels = self.predictions(self._S, lab)
        print((self._labels))
        print((score(lab, self._labels)))
                                                    
    def neg_and_pos(self, files):
        """
        Lists of files according to emotion label (0 for negative, 1 for positive)
        :param files: filenames of inputs     
        :returns:                                         
          - negative_emotion_files: files with negative emotional value (emotional meaning according to the whole performance)              
          - positive_emotion_files: files with positive emotional value (emotional meaning according to the whole performance)   
        """
        for n in range(self.n_class):
            for i, x in enumerate(self._labels):
                if x == n:
                    yield files[i], x
                    
    def save_decisions(self):
        """
        Save the decision_function result in a text file so you can use it in applications
        """   
        array = pandas.DataFrame(self.decision)
        array.to_csv(self.output_dir + '/data.csv') 

    def save_classes(self, files):
        """
        Save the emotions of each sound in a text file so you can use it in applications           
        """ 
        array = pandas.DataFrame(self.neg_and_pos(files))
        array.to_csv(self.output_dir + '/files_classes.csv')            

#create emotions directory in data dir if multitag classification has been performed
def emotions_data_dir(files_dir):
    """                                                                                     
    create emotions directory for all data                                          
    """                                                
    if not os.path.exists(files_dir+'/emotions/happy'):
        os.makedirs(files_dir+'/emotions/happy')
    if not os.path.exists(files_dir+'/emotions/sad'):
        os.makedirs(files_dir+'/emotions/sad')
    if not os.path.exists(files_dir+'/emotions/angry'):
        os.makedirs(files_dir+'/emotions/angry')
    if not os.path.exists(files_dir+'/emotions/relaxed'):
        os.makedirs(files_dir+'/emotions/relaxed')
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
tags_dirs = lambda files_dir: [os.path.join(files_dir,dirs) for dirs in next(os.walk(os.path.abspath(files_dir)))[1] if not os.path.join(files_dir, dirs) == files_dir +'/descriptores']

#emotions dictionary directory (to use with RedPanal API)
def multitag_emotions_dictionary_dir():
    """                                                                                     
    create emotions dictionary directory                                        
    """           
    os.makedirs('data/emotions_dictionary')

#locate all files in data emotions dir
def multitag_emotions_dir(tags_dirs, files_dir, generator):
    """                                                                                     
    locate all files in folders according to multitag emotions classes                                

    :param tags_dirs: directories of tags in data                                                                                            
    :param files_dir: main directory where to save files
    :param generator: generator containing the files (use neg_and_pos)                                                                                                               
    """                                                                                         
    files_format = ['.mp3', '.ogg', '.undefined', '.wav', '.mid', '.wma', '.amr']
    
    emotions_folder = ["/emotions/angry/", "/emotions/sad", "/emotions/relaxed", "/emotions/happy"]
    
    emotions = ["anger", "sadness", "relaxation", "happiness"]                                                                 

    for t, c in list(generator):
        for tag in tags_dirs:
             for f in (list(os.walk(tag, topdown = False)))[-1][-1]:
                 if t.split('.')[0] == f.split('.')[0]:
                     if not f in list(os.walk(str().join((files_dir,emotions_folder[c])), topdown=False)):
                         shutil.copy(os.path.join(tag, f), os.path.join(files_dir+emotions_folder[c], f))     
                         print((str().join((str(f),' evokes ',emotions[c]))))
                         break

from transitions import Machine
import random
import subprocess

#Johnny, Music Emotional State Machine
class MusicEmotionStateMachine(object):
    def __init__(self, name, files_dir):
        self.states = ['angry','sad','relaxed','happy','not angry','not sad', 'not relaxed','not happy']
        self.name = name
        self.state = lambda: random.choice(self.states)
    def sad_music_remix(self, neg_arous_dir, files, decisions, files_dir, harmonic = None):
        for subdirs, dirs, sounds in os.walk(neg_arous_dir):   
            fx = random.choice(sounds[::-1])                    
            fy = random.choice(sounds[:])                      
        x = read(neg_arous_dir + '/' + fx)[0]  
        y = read(neg_arous_dir + '/' + fy)[0] 
        x = (mono_stereo(x) if len(x) == 2 else x) 
        y = (mono_stereo(y) if y.shape[1] == 2 else y) 
        fx = fx.split('.')[0]+'.json'                                  
        fy = fy.split('.')[0]+'.json'                                 
        fx = np.where(files == fx)[0]                       
        fy = np.where(files == fy)[0]         
        if harmonic is False or None:                          
            dec_x = get_coordinate(fx, 1, decisions)                                                        
            dec_y = get_coordinate(fy, 1, decisions)
        else:
            dec_x = get_coordinate(fx, 2, decisions)
            dec_y = get_coordinate(fy, 2, decisions)
        x = source_separation(x, 4)[0]  
        x = scratch_music(x, dec_x)                            
        y = scratch_music(y, dec_y)                           
        x, y = same_time(x,y)                                                                       
        negative_arousal_samples = [i/i.max() for i in (x,y)]                                                                       
        negative_arousal_x = np.array(negative_arousal_samples).sum(axis=0)                                                           
        negative_arousal_x = 0.5*negative_arousal_x/negative_arousal_x.max()                                                              
        if harmonic is True:                                   
            return librosa.decompose.hpss(librosa.stft(negative_arousal_x), margin = (1.0, 5.0))[0]                 
        if harmonic is False or harmonic is None:
            interv = hfc_onsets(np.float32(negative_arousal_x))
            steps = overlapped_intervals(interv)
            output = librosa.effects.remix(negative_arousal_x, steps[::-1], align_zeros = False)
            output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 3)
            remix_filename = files_dir+'/emotions/remixes/sad/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix' 
            write_file(remix_filename, 44100, output)
            subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg']) 
    def happy_music_remix(self, pos_arous_dir, files, decisions, files_dir, harmonic = None):
        for subdirs, dirs, sounds in os.walk(pos_arous_dir):   
            fx = random.choice(sounds[::-1])                    
            fy = random.choice(sounds[:])                      
        x = read(pos_arous_dir + '/' + fx)[0]  
        y = read(pos_arous_dir + '/' + fy)[0] 
        x = (mono_stereo(x) if len(x) == 2 else x) 
        y = (mono_stereo(y) if len(y) == 2 else y) 
        fx = fx.split('.')[0]+'.json'                                  
        fy = fy.split('.')[0]+'.json'                                  
        fx = np.where(files == fx)[0]                      
        fy = np.where(files == fy)[0]                
        if harmonic is False or None:                          
            dec_x = get_coordinate(fx, 3, decisions)                                                        
            dec_y = get_coordinate(fy, 3, decisions)
        else:
            dec_x = get_coordinate(fx, 0, decisions)
            dec_y = get_coordinate(fy, 0, decisions)
        x = source_separation(x, 4)[0] 
        x = scratch_music(x, dec_x)                            
        y = scratch_music(y, dec_y)
        x, y = same_time(x,y)  
        positive_arousal_samples = [i/i.max() for i in (x,y)]  
        positive_arousal_x = np.float32(positive_arousal_samples).sum(axis=0) 
        positive_arousal_x = 0.5*positive_arousal_x/positive_arousal_x.max()
        if harmonic is True:
            return librosa.decompose.hpss(librosa.stft(positive_arousal_x), margin = (1.0, 5.0))[0]  
        if harmonic is False or harmonic is None:
            song = sonify(positive_arousal_x, 44100)
            song.mel_bands_global()
            interv = song.bpm()[1]
            steps = overlapped_intervals(np.int32(interv * 44100))
            output = librosa.effects.remix(positive_arousal_x, steps, align_zeros = False)
            output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 4)
            remix_filename = files_dir+'/emotions/remixes/happy/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
            write_file(remix_filename, 44100, np.float32(output))
            subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def relaxed_music_remix(self, neg_arous_dir, files, decisions, files_dir):
        neg_arousal_h = self.sad_music_remix(neg_arous_dir, files, decisions, files_dir, harmonic = True)
        relaxed_harmonic = librosa.istft(neg_arousal_h)
        interv = hfc_onsets(np.float32(relaxed_harmonic))
        steps = overlapped_intervals(interv)
        output = librosa.effects.remix(relaxed_harmonic, steps[::-1], align_zeros = True)
        output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 4)
        remix_filename = files_dir+'/emotions/remixes/relaxed/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
        write_file(remix_filename, 44100, output)
        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def angry_music_remix(self, pos_arous_dir, files, decisions, files_dir):
        pos_arousal_h = self.happy_music_remix(pos_arous_dir, files, decisions, files_dir, harmonic = True)
        angry_harmonic = librosa.istft(pos_arousal_h)
        song = sonify(angry_harmonic, 44100)
        song.mel_bands_global()
        interv = song.bpm()[1]
        steps = overlapped_intervals(interv)
        output = librosa.effects.remix(angry_harmonic, steps, align_zeros = True)
        output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 3)
        remix_filename = files_dir+'/emotions/remixes/angry/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
        write_file(remix_filename, 44100, np.float32(output))
        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def not_happy_music_remix(self, neg_arous_dir, files, decisions, files_dir):
        sounds = []
        for i in range(len(neg_arous_dir)):
            for subdirs, dirs, s in os.walk(neg_arous_dir[i]):                                  
                sounds.append(subdirs + '/' + random.choice(s))
        fx = random.choice(sounds[::-1])
        fy = random.choice(sounds[:])                    
        x = read(fx)[0]
        y = read(fy)[0]  
        x = (mono_stereo(x) if len(x) == 2 else x) 
        y = (mono_stereo(y) if len(y) == 2 else y) 
        fx = fx.split('/')[-1].split('.')[0]+'.json'                                  
        fy = fy.split('/')[-1].split('.')[0]+'.json'                                   
        fx = np.where(files == fx)[0]                     
        fy = np.where(files == fy)[0]                
        dec_x = get_coordinate(fx, choice(list(range(3))), decisions)               
        dec_y = get_coordinate(fy, choice(list(range(3))), decisions)
        x = source_separation(x, 4)[0] 
        x = scratch_music(x, dec_x)                            
        y = scratch_music(y, dec_y)
        x, y = same_time(x, y)
        not_happy_x = np.sum((x,y),axis=0) 
        not_happy_x = 0.5*not_happy_x/not_happy_x.max()
        interv = hfc_onsets(np.float32(not_happy_x))
        steps = overlapped_intervals(interv)
        output = librosa.effects.remix(not_happy_x, steps[::-1], align_zeros = True)
        output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 3)
        remix_filename = files_dir+'/emotions/remixes/not happy/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
        write_file(remix_filename, 44100, output)
        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def not_sad_music_remix(self, pos_arous_dir, files, decisions, files_dir):
        sounds = []
        for i in range(len(pos_arous_dir)):
            for subdirs, dirs, s in os.walk(pos_arous_dir[i]):                                  
                sounds.append(subdirs + '/' + random.choice(s))
        fx = random.choice(sounds[::-1])
        fy = random.choice(sounds[:])                    
        x = read(fx)[0]  
        y = read(fy)[0]  
        x = (mono_stereo(x) if len(x) == 2 else x) 
        y = (mono_stereo(y) if len(y) == 2 else y) 
        fx = fx.split('/')[-1].split('.')[0]+'.json'                            
        fy = fy.split('/')[-1].split('.')[0]+'.json'                                
        fx = np.where(files == fx)[0]                    
        fy = np.where(files == fy)[0]             
        dec_x = get_coordinate(fx, choice([0,2,3]), decisions)               
        dec_y = get_coordinate(fy, choice([0,2,3]), decisions)
        x = source_separation(x, 4)[0] 
        x = scratch_music(x, dec_x)                            
        y = scratch_music(y, dec_y)
        x, y = same_time(x,y)
        not_sad_x = np.sum((x,y),axis=0) 
        not_sad_x = np.float32(0.5*not_sad_x/not_sad_x.max())
        song = sonify(not_sad_x, 44100)
        song.mel_bands_global()
        interv = song.bpm()[1]
        steps = overlapped_intervals(interv)
        output = librosa.effects.remix(not_sad_x, steps[::-1], align_zeros = True)
        output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 4)
        remix_filename = files_dir+'/emotions/remixes/not sad/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
        write_file(remix_filename, 44100, np.float32(output))
        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def not_angry_music_remix(self, neg_arous_dir, files, decisions, files_dir):
        sounds = []
        for i in range(len(neg_arous_dir)):
            for subdirs, dirs, s in os.walk(neg_arous_dir[i]):                                  
                sounds.append(subdirs + '/' + random.choice(s))
        fx = random.choice(sounds[::-1])
        fy = random.choice(sounds[:])                    
        x = read(fx)[0] 
        y = read(fy)[0]
        x = (mono_stereo(x) if len(x) == 2 else x) 
        y = (mono_stereo(y) if len(y) == 2 else y)   
        fx = fx.split('/')[-1].split('.')[0]+'.json'                                 
        fy = fy.split('/')[-1].split('.')[0]+'.json'                                  
        fx = np.where(files == fx)[0]                      
        fy = np.where(files == fy)[0]              
        dec_x = get_coordinate(fx, choice(list(range(1,3))), decisions)               
        dec_y = get_coordinate(fy, choice(list(range(1,3))), decisions)
        x = source_separation(x, 4)[0] 
        x = scratch_music(x, dec_x)                            
        y = scratch_music(y, dec_y)
        x, y = same_time(x,y)
        stft_morph = np.nan_to_num(morph(x,y,512,0.01,0.7))
        interv = hfc_onsets(np.float32(stft_morph))
        steps = overlapped_intervals(interv)
        output = librosa.effects.remix(stft_morph, steps[::-1], align_zeros = False)
        output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 4)
        remix_filename = files_dir+'/emotions/remixes/not angry/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
        write_file(remix_filename, 44100, output)
        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def not_relaxed_music_remix(self, pos_arous_dir, files, decisions, files_dir):
        sounds = []
        for i in range(len(pos_arous_dir)):
            for subdirs, dirs, s in os.walk(pos_arous_dir[i]):                                  
                sounds.append(subdirs + '/' + random.choice(s))
        fx = random.choice(sounds[::-1])
        fy = random.choice(sounds[:])                    
        x = read(fx)[0]  
        y = read(fy)[0]
        x = (mono_stereo(x) if len(x) == 2 else x) 
        y = (mono_stereo(y) if len(y) == 2 else y)  
        fx = fx.split('/')[-1].split('.')[0]+'.json'                                
        fy = fy.split('/')[-1].split('.')[0]+'.json'                       
        fx = np.where(files == fx)[0]                     
        fy = np.where(files == fy)[0]         
        dec_x = get_coordinate(fx, choice([0,1,3]), decisions)               
        dec_y = get_coordinate(fy, choice([0,1,3]), decisions)
        x = source_separation(x, 4)[0] 
        x = scratch_music(x, dec_x)                            
        y = scratch_music(y, dec_y)
        x, y = same_time(x,y)
        stft_morph = np.nan_to_num(morph(x,y,512,0.01,0.7))
        song = sonify(stft_morph, 44100)
        song.mel_bands_global()
        interv = song.bpm()[1]
        steps = overlapped_intervals(interv)
        output = librosa.effects.remix(stft_morph, steps[::-1], align_zeros = False)
        output = librosa.effects.pitch_shift(output, sr = 44100, n_steps = 3)
        remix_filename = files_dir+'/emotions/remixes/not relaxed/'+str(time.strftime("%Y%m%d-%H:%M:%S"))+'multitag_remix'
        write_file(remix_filename, 44100, np.float32(output))
        subprocess.call(["ffplay", "-nodisp", "-autoexit", remix_filename+'.ogg'])
    def remix(self, files, decisions, files_dir):
        while True:
            state = self.state()            
            if state == 'happy':                             
                print(("ENTERING STATE " + state.upper()))                                  
                self.happy_music_remix(files_dir+'/emotions/happy', files, decisions, files_dir, harmonic = None)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper()))                        
            if state == 'sad':                       
                print(("ENTERING STATE " + state.upper()))                                   
                self.sad_music_remix(files_dir+'/emotions/sad', files, decisions, files_dir, harmonic = None)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
            if state == 'angry':                             
                print(("ENTERING STATE " + state.upper()))                                  
                self.angry_music_remix(files_dir+'/emotions/angry', files, decisions, files_dir)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
            if state == 'relaxed':                           
                print(("ENTERING STATE " + state.upper()))                                  
                self.relaxed_music_remix(files_dir+'/emotions/relaxed', files, decisions, files_dir)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
            if state == 'not happy':                         
                print(("ENTERING STATE " + state.upper()))                                  
                self.not_happy_music_remix([files_dir+'/emotions/sad', files_dir+'/emotions/angry', files_dir+'/emotions/relaxed'], files, decisions, files_dir)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
            if state == 'not sad':                           
                print(("ENTERING STATE " + state.upper()))                                  
                self.not_sad_music_remix([files_dir+'/emotions/happy', files_dir+'/emotions/angry', files_dir+'/emotions/relaxed'], files, decisions, files_dir)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
            if state == 'not angry':                         
                print(("ENTERING STATE " + state.upper()))                                  
                self.not_angry_music_remix([files_dir+'/emotions/happy', files_dir+'/emotions/sad', files_dir+'/emotions/relaxed'], files, decisions, files_dir)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
            if state == 'not relaxed':                       
                print(("ENTERING STATE " + state.upper()))                                 
                self.not_relaxed_music_remix([files_dir+'/emotions/happy', files_dir+'/emotions/sad', files_dir+'/emotions/angry'], files, decisions, files_dir)
                print(("ENTERED STATE " + state.upper())) 
                print(("EXITING STATE " + state.upper())) 
                            

Usage = "./MusicEmotionMachine.py [FILES_DIR] [OUTPUT_DIR] [MULTITAG PROBLEM FalseNone/True] [TRANSITION a/r/s]"

def main():  
    if (len(sys.argv) < 5) or (len(sys.argv) == 5  and (sys.argv[3] == 'False' or sys.argv[3] == 'None')):
        print(("\nBad amount of input arguments\n", Usage, "\n"))
        sys.exit(1)

    try:
        files_dir = sys.argv[1]
        output_dir = sys.argv[2]

        if not os.path.exists(files_dir):                         
            raise IOError("Must run MIR analysis") 

        if sys.argv[3] in ('True'):
            if not 'r' in sys.argv[4]:     
                tags_dir = tags_dirs(files_dir)                
                files = descriptors_and_keys(tags_dir, True)._files  
                features = descriptors_and_keys(tags_dir, True)._features
                fscaled = feature_scaling(features)
                print((len(fscaled)))
                del features                 
                labels = KMeans_clusters(fscaled)
                layers = svm_layers()
                permute = [('linear', 'poly'),                      
                            ('linear', 'sigmoid'),                       
                            ('poly', 'sigmoid')]   
                layers.layer_computation(fscaled, labels, permute)
                                       
                labl = layers.best_labels()
                layers.attention() 
                
                permute = [('linear', 'poly'),
                            ('linear', 'rbf'),                      
                            ('linear', 'sigmoid'),                       
                            ('rbf', 'linear'),
                            ('sigmoid', 'linear'),
                            ('sigmoid', 'poly'),
                            ('sigmoid', 'rbf')] 

                layers.layer_computation(layers.attention_dataset, labl, permute)
                labl = layers.best_labels()
                layers.attention() 

                layers.store_attention(files, sys.argv[2])

                fx = layers.attention_dataset  
                layers.store_good_labels(labl, sys.argv[2])

            if 'a' in sys.argv[4]:  
                sys.exit()    

            if 'r' in sys.argv[4]:  
                fx, files = read_attention_file(sys.argv[2])
                labl = read_good_labels(sys.argv[2])
                labl = labl[0][1:]

            if 's' in sys.argv[4] or 'r' in sys.argv[4]:                                        
                msvm = main_svm(fx, labl, 1./0.001, 0.001, 1./fx.shape[1], ['linear', 'poly'], output_dir) 
                try:                          
                    tags_dir                  
                except Exception as e:        
                    tags_dir = tags_dirs(files_dir)
                msvm.save_decisions()         
                msvm.save_classes(files)          
                emotions_data_dir(files_dir)                                   
                multitag_emotions_dir(tags_dir, files_dir, msvm.neg_and_pos(files))

        if (sys.argv[3] in ('None')) or (sys.argv[3] in ('False')): 
            files_dir = sys.argv[1]
            data_dir = sys.argv[2]    
            files, labels, decisions = read_file(data_dir+'/files_classes.csv', data_dir)  
            me = MusicEmotionStateMachine("Johnny", files_dir) #calling Johnny        
            me.remix(files, decisions, files_dir)
    except Exception as e:                     
        logger.exception(e)

if __name__ == '__main__': 
    main()
