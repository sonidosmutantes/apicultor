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
      - scaled features with mean and standard deviation
    """ 
    #test again or normalize accordingly   
    return MinMaxScaler().fit_transform(sc().fit(f).transform(f))

def KMeans_clusters(fscaled):
    """
    KMeans clustering for features                                                           
    :param fscaled: scaled features                                                                                         
    :returns:                                                                                                         
      - labels: classes         
    """
    labels = (KMeans(init = pca(n_components = 4).fit(fscaled).components_, n_clusters=4, n_init=1, precompute_distances = True, random_state = 0, n_jobs = -1).fit(fscaled).labels_)
    return labels

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
            self.scores.append(err)
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
                print('Backward trust compatibility for target',tc,'is',btc[tc]) 
            #compute BEC
            bec, cons = BEC(labels_test, pred_c)   
            for ec in range(len(bec)):
                print('Backward error compatibility for target',ec,'is',bec[ec])  
            if i > 0:
                self.sample_conflicts = []
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
        with open(fdir+'/models/'+signature+'_modelsvs.npy','wb') as file:
            np.save(file, self.best_layer.svs)
        with open(fdir+'/models/'+signature+'_modeltargets.npy','wb') as file:
            np.save(file, self.best_layer.targets_outputs)  
        with open(fdir+'/models/'+signature+'_modelsvs_locs.npy','wb') as file:
            np.save(file, self.best_layer.sv_locs)      
        with open(fdir+'/models/'+signature+'_modeldual_coef.npy','wb') as file:
            np.save(file, self.best_layer.dual_coefficients) 
        with open(fdir+'/models/'+signature+'_modelnclasses.npy','wb') as file:
            np.save(file, self.best_layer.n_class) 
        with open(fdir+'/models/'+signature+'_modelbias.npy','wb') as file:
            np.save(file, self.best_layer.bias) 

    def read_checkpoint(self,fdir,signature):  
        """
        Read model data in a specified directory
        """
        with open(fdir+'/models/'+signature+'_modelsvs.npy','rb') as file:
            self.svs=np.load(file)
        with open(fdir+'/models/'+signature+'_modeltargets.npy','rb') as file:
            self.best_labels=np.load(file)  
        with open(fdir+'/models/'+signature+'_modelsvs_locs.npy','rb') as file:
            self.sv_locs=np.load(file)      
        with open(fdir+'/models/'+signature+'_modeldual_coef.npy','rb') as file:
            self.dual_coefficients=np.load(file)
        with open(fdir+'/models/'+signature+'_modelnclasses.npy','rb') as file:
            self.n_classes=np.load(file)
        with open(fdir+'/models/'+signature+'_modelbias.npy','rb') as file:
            self.bias=np.load(file)             

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
