#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from ..utils.dj import *
from ..machine_learning.cross_validation import *
from ..gradients.descent import SGD
from ..constraints.bounds import dsvm_low_a as la
from ..constraints.bounds import dsvm_high_a as ha
from ..constraints.bounds import es
from ..constraints.tempo import same_time
from ..constraints.time_freq import *
from ..gradients.subproblem import *
from ..utils.data import get_files, get_dics, desc_pair, read_file
from ..utils.algorithms import morph
import time
from collections import Counter, defaultdict
from itertools import combinations
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
from sklearn.decomposition.pca import PCA as pca
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
      - scaled features with mean and standard deviation
    """    
    return MinMaxScaler().fit_transform(sc().fit(f).transform(f))

def KMeans_clusters(fscaled):
    """
    KMeans clustering for features                                                           
    :param fscaled: scaled features                                                                                         
    :returns:                                                                                                         
      - labels: classes         
    """
    labels = (KMeans(init = pca(n_components = 4).fit(fscaled).components_, n_clusters=4, n_init=1, precompute_distances = True).fit(fscaled).labels_)
    return labels

class deep_support_vector_machines(object):
    """
    Functions for Deep Support Vector Machines                                               
    :param features: scaled features                                                                                        
    :param labels: classes                                                                                            
    """ 
    def __init__(self, kernel1, kernel2):
        self.kernel1 = None
        self.kernel2 = None
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

        sk = ssd(x, y.T, dense_output = True)

        sk *= gamma

        sk += c

        np.tanh(np.array(sk, dtype='float64'), np.array(sk, dtype = 'float64'))
        return np.array(sk, dtype = 'float64')
    @classmethod
    @memoize
    def rbf_kernel(self, x, y, gamma):
        """
        Custom radial basis function, similarities of vectors using a radial basis function kernel
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
            self.instances = list(combinations(range(self.n_class), 2))
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

            iterations = 0                                            
            diff = 1. + reg_param 
            for i in range(int(n_f/2)):
                a0 = a.copy()
                p1 = []
                p2 = []                               
                for k in range(n_f):
                    zero = int(uniform(k, n_f0-1))
                    one = int(uniform(n_f0, n_f))                        
                    if (not zero in p1) and (not one in p2):                                             
                        p1.append(zero)                                        
                        p2.append(one)
                    if (not zero >= k):                                             
                        break
                for j in range(len(p1)): 
                    g1 = g(lab[p1[j]], a0[p1[j]], self.Q[p1[j],p1[j]])
                    g2 = g(lab[p2[j]], a0[p2[j]], self.Q[p2[j],p2[j]])
                    w0 = es(a, lab, self.trained_features[c]) 
                    direction_grad = 0.8 * ((w0 * g2) + ((1 - w0) * g1)) #step with 0.8 of learning rate

                    if (direction_grad * (g1 + g2) <= 0.):        
                        break

                    a[p1[j]] = a0[p1[j]] + direction_grad #do step
                    a[p2[j]] = a0[p2[j]] + direction_grad #do step

                a = la(a) #satisfy constraints for lower As
                a = ha(a, class_weight[self.classesm[c]], self.C) #satisfy constraints for higher As using class weights
                diff = Q_a(a, self.Q) - Q_a(a0, self.Q)
                iterations += 1
                if (diff < reg_param):
                    print(str().join(("Gradient Ascent Converged after ", str(iterations), " iterations")))
                    break

            #alphas are automatically signed and those with no incidence in the hyperplane are 0 complex or -0 complex, so alphas != 0 are taken 
            a = SGD(a, lab, self.Q * lab, reg_param)
            a = ha(a, class_weight[self.classesm[c]], self.C) #keep up with the penalty boundary
            a = ha(a * -1, class_weight[self.classesm[c]], self.C)
            a *= -1
                              
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

        a = defaultdict(list)
        for i, (j,m) in enumerate(self.instances):
            a[j].append(self.a[i][self.a[i] < 0])
            a[m].append(self.a[i][self.a[i] > 0])

        ns = defaultdict(list)
        for i, (j,m) in enumerate(self.instances):
            ns[j].append(self.ns[i][0])
            ns[m].append(self.ns[i][1])

        nsv = [np.hstack(list(ns.values())[j][i] for i in range(len(list(ns.values())[j]))) for j in range(len(list(ns.values())))]

        unqs = np.array([np.unique(nsv[i], return_index = True, return_counts = True) for i in range(len(nsv))])

        uniques = [list(unqs[:,0][i]) for i in range(len(unqs[:,0]))]

        counts = [list(unqs[:,2][i]) for i in range(len(unqs[:,0]))]

        svs_ = [list(np.array(counts[i]) == self.n_class - 1) for i in range(len(counts))]

        self.ns = [uniques[i][j] for i in range(len(uniques)) for j in range(len(uniques[i])) if counts[i][j] == (self.n_class - 1)]

        for i in range(len(a)):           
            for j in range(len(a[i])):
                for m in range(len(a[i][j])):
                    if ns[i][j][m] in self.ns:
                        self.dual_coefficients[j].append(a[i][j][m])

        self.dual_coefficients = np.array(self.dual_coefficients)

        self.nvs = [len(np.array(uniques[i])[np.array(svs_[i])]) for i in range(len(uniques))]

        self.svs = features[self.ns] #update support vectors

        self.sv_locs = np.cumsum(np.hstack([[0], self.nvs]))

        self.w = [] #update weights given common support vectors, the other values helped making sure it wasn't restarting

        for class1 in range(self.n_class):
            # SVs for class1:
            sv1 = self.svs[self.sv_locs[class1]:self.sv_locs[class1 + 1], :]
            for class2 in range(class1 + 1, self.n_class):
                # SVs for class1:
                sv2 = self.svs[self.sv_locs[class2]:self.sv_locs[class2 + 1], :]
                # dual coef for class1 SVs:
                alpha1 = self.dual_coefficients[class2 - 1, self.sv_locs[class1]:self.sv_locs[class1 + 1]]
                # dual coef for class2 SVs:
                alpha2 = self.dual_coefficients[class1, self.sv_locs[class2]:self.sv_locs[class2 + 1]]
                # build weight for class1 vs class2
                self.w.append(ssd(alpha1, sv1)
                            + ssd(alpha2, sv2))

        if self.kernel1 == "poly":
            kernel1 = self.polynomial_kernel
        if self.kernel1 == "sigmoid":
            kernel1 = self.sigmoid_kernel
        if self.kernel1 == "rbf":
            kernel1 = self.rbf_kernel 
        if self.kernel1 == "linear":
            kernel1 = self.linear_kernel_matrix

        self.bias = []
        for class1 in range(self.n_class):
            sv1 = self.svs[self.sv_locs[class1]:self.sv_locs[class1 + 1], :]
            for class2 in range(class1 + 1, self.n_class):
                sv2 = self.svs[self.sv_locs[class2]:self.sv_locs[class2 + 1], :]
                if kernel1 == self.linear_kernel_matrix:
                    self.bias.append(-((kernel1(sv1, self.w[class1].T).max() + kernel1(sv2, self.w[class2].T).min())/2))
                else:
                    self.bias.append(-((kernel1(sv1, self.w[class1], self.gamma).max() + kernel1(sv2, self.w[class2], self.gamma).min())/2))
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
            self.k = self.svs @ features.T
        else:           
            self.k = kernel(self.svs, features, self.gamma)     

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
        
        self.__these_predicted = np.copy(targts)

        votes = np.zeros((len(features), self.n_class))

        for i in self.classes:
            for j in range(i + 1, len(self.classes)):
                votes[:,i][self.decision[:,i] > 0] += 1
                votes[:,j][self.decision[:,j] < 0] += 1 

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
        self.fxs = []
        self.targets_outputs = []
        self.scores = []
        sample_weight = float(len(features)) / (len(np.array(list(set(labels)))) * np.bincount(labels))
        Cs = [1./0.01, 1./0.04, 1./0.06, 1./0.08, 1./0.1, 1./0.12]
        reg_params = [0.01, 0.04, 0.06, 0.08, 0.1, 0.12]
        self.kernel_configs = [['linear', 'poly'], ['linear', 'rbf'], ['linear', 'sigmoid'], ['poly', 'sigmoid'], ['poly', 'rbf'], ['rbf', 'sigmoid']]
        for i in range(len(self.kernel_configs)):
            print ("Calculating values for prediction")
            best_estimator = GridSearch(self, features, labels, Cs, reg_params, [self.kernel_configs[i]])
            print(("Best estimators are: ") + str(best_estimator['C'][0]) + " for C and " + str(best_estimator['reg_param'][0]) + " for regularization parameter")
            self.fit_model(features, labels, self.kernel_configs[i][0], self.kernel_configs[i][1], best_estimator['C'][0][0], best_estimator['reg_param'][0][0], 1./features.shape[1])
            print ("Predicting")
            pred_c = self.predictions(features, labels)
            print(("Predicted"), pred_c)
            err = score(labels, pred_c)
            self.scores.append(err)
            if err < 0.5:
                self.fxs.append(self.decision)            
                self.targets_outputs.append(pred_c)
        if len(self.fxs) <= 1:
            raise IndexError("You've been giving bad information. Most likely data that doesn't scale very good enough")

        return self

    def store_attention(self, files, output_dir):
        """
        Save the decision_function result in a text file so you can use it in applications
        """  
        d = defaultdict(list) 
        for i in range(len(files)):
            d[files[i]] = self.attention_dataset[i].tolist()    
        array = pandas.DataFrame(d)
        array.to_csv(output_dir + '/attention.csv')  

    def attention(self):
        """
        A function that averages feasibility in the model                         
        """
        class_weight = float(len(self.targets))/(4 * np.bincount(self.targets)) 
        w = np.broadcast_to(class_weight, (np.array(self.fxs).T.ndim-1)*(1,) + class_weight.shape)
        w = w.swapaxes(-1, 2)
        self.attention_dataset = np.multiply(np.array(self.fxs), w).sum(axis=0)/(sum(w))
        return self

    def best_labels(self):
        """
        Get the labels with highest output in the input layers instances       
        """ 
        self.targets_outputs = np.int32([Counter(np.int32(self.targets_outputs)[:,i]).most_common()[0][0] for i in range(np.array(self.targets_outputs).shape[1])])

        return self.targets_outputs 

    def store_good_labels(self, files, output_dir):
        """
        Save the decision_function result in a text file so you can use it in applications
        """  
        d = defaultdict(list) 
        for i in range(len(files)):
            d[files[i]] = self.targets_outputs[i] 
        array = pandas.DataFrame(d)
        array.to_csv(output_dir + '/attention_labels.csv')  

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
        self.fit_model(self._S, lab, kernels_config[0], kernels_config[1], C, reg_param, gamma)
        self._labels = self.predictions(self._S, lab)
        print(self._labels)
        print(score(lab, self._labels))
                                                    
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
                         print(str().join((str(f),' evokes ',emotions[c])))

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
        y = (mono_stereo(y) if len(y) == 2 else y) 
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
                print("ENTERING STATE " + state.upper())                                  
                self.happy_music_remix(files_dir+'/emotions/happy', files, decisions, files_dir, harmonic = None)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper())                        
            if state == 'sad':                       
                print("ENTERING STATE " + state.upper())                                   
                self.sad_music_remix(files_dir+'/emotions/sad', files, decisions, files_dir, harmonic = None)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
            if state == 'angry':                             
                print("ENTERING STATE " + state.upper())                                  
                self.angry_music_remix(files_dir+'/emotions/angry', files, decisions, files_dir)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
            if state == 'relaxed':                           
                print("ENTERING STATE " + state.upper())                                  
                self.relaxed_music_remix(files_dir+'/emotions/relaxed', files, decisions, files_dir)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
            if state == 'not happy':                         
                print("ENTERING STATE " + state.upper())                                  
                self.not_happy_music_remix([files_dir+'/emotions/sad', files_dir+'/emotions/angry', files_dir+'/emotions/relaxed'], files, decisions, files_dir)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
            if state == 'not sad':                           
                print("ENTERING STATE " + state.upper())                                  
                self.not_sad_music_remix([files_dir+'/emotions/happy', files_dir+'/emotions/angry', files_dir+'/emotions/relaxed'], files, decisions, files_dir)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
            if state == 'not angry':                         
                print("ENTERING STATE " + state.upper())                                  
                self.not_angry_music_remix([files_dir+'/emotions/happy', files_dir+'/emotions/sad', files_dir+'/emotions/relaxed'], files, decisions, files_dir)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
            if state == 'not relaxed':                       
                print("ENTERING STATE " + state.upper())                                 
                self.not_relaxed_music_remix([files_dir+'/emotions/happy', files_dir+'/emotions/sad', files_dir+'/emotions/angry'], files, decisions, files_dir)
                print("ENTERED STATE " + state.upper()) 
                print("EXITING STATE " + state.upper()) 
                            

Usage = "./MusicEmotionMachine.py [FILES_DIR] [OUTPUT_DIR] [MULTITAG PROBLEM FalseNone/True] [TRANSITION a/r/s]"

def main():  
    if (len(sys.argv) < 5) or (len(sys.argv) == 5  and (sys.argv[3] == 'False' or sys.argv[3] == 'None')):
        print("\nBad amount of input arguments\n", Usage, "\n")
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
                print (len(fscaled))
                del features                 
                labels = KMeans_clusters(fscaled)
                layers = svm_layers()                
                layers.layer_computation(fscaled, labels)
                                       
                labl = layers.best_labels()

                layers.attention() 
                layers.store_attention(files, sys.argv[2])

                fx = layers.attention_dataset  
                layers.store_good_labels(labl, sys.argv[2])

            if 'a' in sys.argv[4]:  
                sys.exit()    

            if 'r' in sys.argv[4]:  
                fx = read_attention_file(sys.argv[2])
                labl = read_good_labels(sys.argv[2])

            if 's' in sys.argv[4] or 'r' in sys.argv[4]:                             
                Cs = [1./0.1, 1./0.33, 1./0.4, 1./0.6, 1./0.8] #it should work well with less parameter searching                           
                reg_params = [0.1, 0.33, 0.4, 0.6, 0.8]  #add 0.1, it might work in cases with a smaller dataset                                
                kernel_configs = [['linear', 'poly'], ['linear', 'rbf'], ['linear', 'sigmoid']] #Also can use rbf with linear if you've got difficult to handle data, or try your parameters                                                                   
                best_estimators = GridSearch(svm_layers(), fx, labl, Cs, reg_params, kernel_configs)          
                C, reg_param, kernels_config = best_kernels_output(best_estimators, kernel_configs)                                            
                msvm = main_svm(fx, labl, C[0], reg_param[0], 1./fx.shape[1], kernels_config, output_dir) 
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
