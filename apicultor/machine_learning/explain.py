import numpy as np
from sklearn.metrics import  accuracy_score
from random import sample
import logging
import warnings

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def acc_score(targs, classes):
    """
    return an accuracy score of how well predictions did
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """  
    lw = np.ones(len(targs))
    for idx, m in enumerate(np.bincount(targs)):            
        lw[targs == idx] *= (m/float(targs.shape[0]))
    return accuracy_score(targs, classes, sample_weight = lw)

#adversarial method
def explain(model,x,ys,intersects,idxs):
    """
    Since the number of relevant features in a dataset must be less than the number of  targets,
    this method follows given criteria to split the dataset into subsets to predict accuracy 
    the hypothetic way (for a expected feature criteria accuracies and relevant  features must be given
    without using the complete dataset)
    :param model (type(model) == object): a usable model
    :param x (type(x) == np.darray): dataset that is going to be explained
    :param ys (type(ys) == np.darray): related targets
    :param intersects (type(intersects) == list): list of ascendent criteria (type(criteria) == bool) < number 
        of targets (eg.: [x[:,0] > 900, x[:,4] > 70])
    :param idxs (type(intersects) == list): indexes of important features to perturb
    :returns:                                                                                                         
      - pos_explanation_scores: accuracies for data following the instances criterias
      - neg_explanation_scores: accuracies for data not following the instances criterias
    """
    #0vsall -> 
    #1vsall ->
    pos_explanation_scores = []
    neg_explanation_scores = []
    truek = model.kernel1   
    model.kernel1 = 'rbf'
    noisy_x = x.copy()
    for yi in range(len(intersects)):    	
        targets = model.predictions(noisy_x,ys)
        noisy_x[:,idxs[yi]] = np.array(sample(list(noisy_x[:,idxs[yi]]),noisy_x[:,idxs[yi]].size))
        try:
            pos_explanation_scores.append(acc_score(ys[np.where(noisy_x[:,idxs[yi]]>intersects[yi])],targets[np.where(noisy_x[:,idxs[yi]]>intersects[yi])]))
            neg_explanation_scores.append(acc_score(ys[np.where(noisy_x[:,idxs[yi]]<intersects[yi])],targets[np.where(noisy_x[:,idxs[yi]]<intersects[yi])]))
            noisy_x = noisy_x[np.where(noisy_x[:,idxs[yi]]>intersects[yi])]
            ys = ys[np.where(noisy_x[:,idxs[yi]]>intersects[yi])]
        except Exception as e:
            print(logger.exception(e))
    model.kernel1 = truek
    #add code for effect size or feature transformation    
    #given expected criteria, return accuracies per instance
    return pos_explanation_scores, neg_explanation_scores
