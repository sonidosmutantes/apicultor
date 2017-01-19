#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, mean_squared_error
from itertools import product
import numpy as np
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  

def score(targs, classes):
    """
    return an accuracy score of how well predictions did
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the accuracy score (higher than 0.5 should be a good indication)
    """  
    lw = np.ones(len(targs))
    for idx, m in enumerate(np.bincount(targs)):            
        lw[targs == idx] *= (m/float(targs.shape[0]))
    return mean_squared_error(targs, classes, sample_weight = lw)

# it's going to work with MEM DSVM implementation
def GridSearch(model, features, targets, Cs, reg_params, kernel_configs):
    n = len(features)               
    """
    Perform Cross-Validation using search to find out which is the best configuration for a layer. Ideally a full cross-validation process would need at least +4000 configurations to find a most suitable, but here we are setting parameters rationally to set accurate values. The MEM uses an automatic value for gamma of 1./n_features.  
    :param model: predictive targets
    :param features: predicted targets
    :param targets: predictive targets
    :param Cs: list of C configurations
    :param reg_params: list of reg_param configurations
    :param kernel_configs: kernels configurations
    :returns:                                                                                                         
      - the best estimator values (its accuracy score, its C and its reg_param value)
    """  
    test_n = n/10 # 1/10 of data to test 
    train_n = n - test_n 
    features_train, targets_train, features_test, targets_test = features[:train_n], targets[:train_n], features[train_n:], targets[train_n:]                                
    from collections import defaultdict               
    params = defaultdict(list)
    timing = defaultdict(list)      
    scores = defaultdict(list) 
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    best_estimator = defaultdict(list) 
    [best_estimator['score'].append([]) for i in range(len(kernel_configs))] 
    [best_estimator['C'].append([]) for i in range(len(kernel_configs))] 
    [best_estimator['reg_param'].append([]) for i in range(len(kernel_configs))] 
    [scores['scorings'].append([]) for i in range(len(kernel_configs))] 
    [train_scores['scorings'].append([]) for i in range(len(kernel_configs))]     
    [test_scores['scorings'].append([]) for i in range(len(kernel_configs))] 
    [timing['training_times'].append([]) for i in range(len(kernel_configs))]
    [timing['classifier times'].append([]) for i in range(len(kernel_configs))]              
    params['C'].append(Cs)        
    params['reg_param'].append(reg_params)
    for i in range(len(kernel_configs)):
        for j in range(len(params['C'][0])):
            try:                                                   
                training_time = time.time()                            
                clf_train = model.fit_model(features_train, targets_train, kernel_configs[i][0], kernel_configs[i][1], params['C'][0][j], params['reg_param'][0][j], 1./features.shape[1])                          
                training_time = np.abs(training_time - time.time())            
                timing['training_times'][i].append(training_time)         
                print ("Training time is: " + str(training_time))      
                clf_predictions_train = model.predictions(features_train, targets_train)                    
                train_scores['scorings'][i].append(score(targets_train, clf_predictions_train))                                                  
                print ("Train Scoring Error is: " + str(train_scores['scorings'][i][j])) 
                clf_test = model.fit_model(features_test, targets_test, kernel_configs[i][0], kernel_configs[i][1], params['C'][0][j], params['reg_param'][0][j], 1./features.shape[1])                                                                      
                clf_predictions_test = model.predictions(features_test, targets_test)                       
                test_scores['scorings'][i].append(score(targets_test, clf_predictions_test))                                                     
                print ("Test Scoring Error is: " + str(test_scores['scorings'][i][j]))                                                                 
                clf_time = time.time()                                 
                clf = model.fit_model(features, targets, kernel_configs[i][0], kernel_configs[i][1], params['C'][0][j], params['reg_param'][0][j], 1./features.shape[1])                                            
                clf_time = np.abs(clf_time - time.time())                      
                timing['classifier times'][i].append(clf_time)            
                print ("Classification time is: " + str(clf_time))     
                clf_predictions = model.predictions(features, targets)          
                scores['scorings'][i].append(score(targets, clf_predictions))                                                                    
                print ("Scoring error is: " + str(scores['scorings'][i][j]))    
            except Exception, e:                                       
                print logger.exception(e)
        best_estimator['score'][i].append(min(scores['scorings'][i]))
        best_estimator['C'][i].append(params['C'][0][np.array(scores['scorings'][i]).argmin()])
        best_estimator['reg_param'][i].append(params['reg_param'][0][np.array(scores['scorings'][i]).argmin()])
        print (("Best estimators are: ") + ("C: ") + str(best_estimator['C'][i]) + ("Regularization parameter: ") + str(best_estimator['reg_param'][i])) 
        print (("Mean training error scorings: ") + str(np.mean(train_scores['scorings'][i])))
        print (("std training error scorings: ") + str(np.mean(train_scores['scorings'][i])))
        print (("Mean test error scorings: ") + str(np.std(test_scores['scorings'][i])))
        print (("std test error scorings: ") + str(np.std(test_scores['scorings'][i])))
        print (("Mean error scorings: ") + str(np.mean(scores['scorings'][i])))
        print (("std error scorings: ") + str(np.std(scores['scorings'][i])))
    return best_estimator
