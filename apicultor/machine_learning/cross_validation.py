#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import logging
from .fairness import *

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
    features_train = []
    features_test = []
    targets_train = []
    targets_test = []
    for i in range(len(np.unique(targets))):
        n = len(features[targets==i])
        test_n = int((20 * n)/100)
        train_n = n - test_n
        features_train.append(features[targets==i][:train_n])
        features_test.append(features[targets==i][train_n:])
        targets_train.append(targets[targets==i][:train_n])
        targets_test.append(targets[targets==i][train_n:])
    features_test = np.vstack(features_test)
    features_train = np.vstack(features_train)
    targets_train = np.hstack(targets_train)
    targets_test = np.hstack(targets_test)
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
                model.fit_model(features_train, targets_train, kernel_configs[i][0], kernel_configs[i][1], params['C'][0][j], params['reg_param'][0][j], 1./features_train.shape[1], 0.8)                          
                training_time = np.abs(training_time - time.time())            
                timing['training_times'][i].append(training_time)         
                print(str().join(("Training time is: ", str(training_time))))      
                clf_predictions_train = model.predictions(features_train, targets_train)                    
                train_scores['scorings'][i].append(score(targets_train, clf_predictions_train)) 
                for inst in model.instances: 
                    y_pred = np.hstack((clf_predictions_train[targets_train==inst[0]],clf_predictions_train[targets_train==inst[1]])) 
                    y = np.hstack((targets_train[targets_train==inst[0]],clf_predictions_train[targets_train==inst[1]])) 
                    protection_val_rule_train = p_rule(y_pred,y,inst[0],inst[1]) 
                    if type(protection_val_rule_train) == float and protection_val_rule_train >= .8: 
                        pass             
                    else: 
                        print(ValueError("Discrimination and false information at training instance"))
                        continue          
                print(str().join(("Train Scoring Error is: ", str(train_scores['scorings'][i][j]))))
                clf_predictions_test = model.predictions(features_test, targets_test)                      
                test_scores['scorings'][i].append(score(targets_test, clf_predictions_test))       
                for inst in model.instances: 
                    y_pred = np.hstack((clf_predictions_test[targets_test==inst[0]],clf_predictions_test[targets_test==inst[1]])) 
                    y = np.hstack((targets_test[targets_test==inst[0]],clf_predictions_test[targets_test==inst[1]])) 
                    protection_val_rule_test = p_rule(y_pred,y,inst[0],inst[1])                                                             
                    if type(protection_val_rule_test) == float and protection_val_rule_test >= .8: 
                        pass             
                    else: 
                        print(ValueError("Discrimination and false information at test instance"))
                        continue
                print(str().join(("Test Scoring Error is: ", str(test_scores['scorings'][i][j]))))      
                clf_time = time.time()                                 
                model.fit_model(features, targets, kernel_configs[i][0], kernel_configs[i][1], params['C'][0][j], params['reg_param'][0][j], 1./features.shape[1], 0.8)                                            
                clf_time = np.abs(clf_time - time.time())                      
                timing['classifier times'][i].append(clf_time)            
                print(str().join(("Classification time is: ", str(clf_time))))     
                clf_predictions = model.predictions(features, targets)            
                for inst in model.instances: 
                    y_pred = np.hstack((clf_predictions[targets==inst[0]],clf_predictions[targets==inst[1]])) 
                    y = np.hstack((targets[targets==inst[0]],clf_predictions[targets==inst[1]])) 
                    protection_val_rule = p_rule(y_pred,y,inst[0],inst[1]) 
                    print("Non protected group is ",inst[1]," and protected group is ",inst[0])                         
                    if type(protection_val_rule) == float and protection_val_rule >= .8: 
                        pass             
                    else: 
                        print(ValueError("Discrimination and false information at classification"))
                        print("Current error is :",protection_val_rule)       
                        continue            
                scores['scorings'][i].append(score(targets, clf_predictions))                                                                    
                print(str().join(("Scoring error is: ", str(scores['scorings'][i][j]))))    
            except Exception as e:                                       
                print("Something happened! Continuing...",e)
                if len(train_scores['scorings'][i])-1 == j: train_scores['scorings'][i].pop(j)
                if len(train_scores['scorings'][i])-1 != j: train_scores['scorings'][i].append(2)
                if len(test_scores['scorings'][i])-1 == j: test_scores['scorings'][i].pop(j)
                if len(test_scores['scorings'][i])-1 != j: test_scores['scorings'][i].append(2)
                if len(scores['scorings'][i])-1 == j: scores['scorings'][i].pop(j)
                if len(scores['scorings'][i])-1 != j: scores['scorings'][i].append(2)
        best_estimator['score'][i].append(min(scores['scorings'][i]))
        best_estimator['C'][i].append(params['C'][0][np.array(scores['scorings'][i]).argmin()])
        best_estimator['reg_param'][i].append(params['reg_param'][0][np.array(scores['scorings'][i]).argmin()])
        print(str().join(("Best estimators are: ", "C: ", str(best_estimator['C'][i]), " Regularization parameter: ", str(best_estimator['reg_param'][i])))) 
        print(str().join(("Mean training error scorings: ", str(np.mean(train_scores['scorings'][i])))))
        print(str().join(("std training error scorings: ", str(np.mean(train_scores['scorings'][i])))))
        print(str().join(("Mean test error scorings: ", str(np.std(test_scores['scorings'][i])))))
        print(str().join(("std test error scorings: ", str(np.std(test_scores['scorings'][i])))))
        print(str().join(("Mean error scorings: ", str(np.mean(scores['scorings'][i])))))
        print(str().join(("std error scorings: ", str(np.std(scores['scorings'][i])))))
    return best_estimator
