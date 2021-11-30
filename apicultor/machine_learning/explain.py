import numpy as np
from sklearn.metrics import  accuracy_score
from .visuals import plot_regression
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
#productive
def explain(model,x,ys,limit,idxs,logical=None,plot=None,fig=None,thresh=None):
    """
    Since the number of relevant features in a dataset must be less than the number of  targets,
    this method follows given criteria to split the dataset into subsets to predict accuracy 
    the hypothetic way (for a expected feature criteria accuracies and relevant  features must be given
    without using the complete dataset)
    :param model (type(model) == object): a usable model
    :param x (type(x) == np.darray): dataset that is going to be explained
    :param ys (type(ys) == np.darray): related targets
    :param limit (type(limit) == list): if a list of strings is given only 'mean', 'var' or 'std' are supported,
        it also supports other types. This argument sets the intersection value (eg.: [x[:,0] > 900, x[:,4] > 70])
    :param idxs (type(intersects) == list): indexes of important features to perturb
    :param logical (type(logical) == bool or type(logical) == list): bool or a list of bools (must be of the size of limits).
        If True is passed, observability is based on positive limit. If False is passed, observability is based on not
        positive limit. If None is passed, the current limit is not observable and therefore it is contrasted against the 
        explaining values and the next limit is used against the explaining data (limit_i < data < limit_i+1)  	
    :param thresh: a threshold for robust explanation
    :returns:                                                                                                         
      - pos_explanation_scores: accuracies for data following the instances criterias
      - neg_explanation_scores: accuracies for data not following the instances criterias
      - plt: a plot of the explanation
    """
    #0vsall -> 
    #1vsall ->
    pos_explanation_scores = []
    neg_explanation_scores = []
    truek = model.kernel1   
    model.kernel1 = 'rbf'
    noisy_x = x.copy()
    for yi in range(len(limit)):    	
        targets = model.predictions(noisy_x,ys)
        noisy_x[:,idxs[yi]] = np.array(sample(list(noisy_x[:,idxs[yi]]),noisy_x[:,idxs[yi]].size))
        if limit[yi] == 'mean':
            intersection = noisy_x[:,idxs[yi]].mean()
            #intersection = noisy_x[:,idxs[yi]].mean()*thresh
        elif limit[yi] == 'var':
            intersection = np.var(noisy_x[:,idxs[yi]])
            #intersection = np.var(noisy_x[:,idxs[yi]])*thresh
        elif limit[yi] == 'std':
            intersection = np.std(noisy_x[:,idxs[yi]])
            #intersection = np.std(noisy_x[:,idxs[yi]])*thresh
        else:
            intersection = limit[yi]
        try:
            if logical != None and logical != False:
                if len(logical) != len(limit):
                    raise ValueError('Missing logic')
                if logical[yi] != True:
                    pos_explanation_scores.append(acc_score(ys[np.where(np.logical_not(noisy_x[:,idxs[yi]]>intersection))],targets[np.where(np.logical_not(noisy_x[:,idxs[yi]]>intersection))]))
                    neg_explanation_scores.append(acc_score(ys[np.where(np.logical_not(noisy_x[:,idxs[yi]]<intersection))],targets[np.where(np.logical_not(noisy_x[:,idxs[yi]]<intersection))]))
                    #data drift
                    noisy_x = noisy_x[np.where(np.logical_not(noisy_x[:,idxs[yi]]>intersection))]
                    #concept drift
                    ys = ys[np.where(np.logical_not(noisy_x[:,idxs[yi]]>intersection))]
                else:               
                    pos_explanation_scores.append(acc_score(ys[np.where(noisy_x[:,idxs[yi]]>intersection)],targets[np.where(noisy_x[:,idxs[yi]]>intersection)]))
                    neg_explanation_scores.append(acc_score(ys[np.where(noisy_x[:,idxs[yi]]<intersection)],targets[np.where(noisy_x[:,idxs[yi]]<intersection)]))
                    #data drift
                    noisy_x = noisy_x[np.where(noisy_x[:,idxs[yi]]>intersection)]
                    #concept drift
                    ys = ys[np.where(noisy_x[:,idxs[yi]]>intersection)]
            else:
                pos_explanation_scores.append(acc_score(ys[np.where(noisy_x[:,idxs[yi]]>intersection)],targets[np.where(noisy_x[:,idxs[yi]]>intersection)]))
                neg_explanation_scores.append(acc_score(ys[np.where(noisy_x[:,idxs[yi]]<intersection)],targets[np.where(noisy_x[:,idxs[yi]]<intersection)]))
                #data drift
                noisy_x = noisy_x[np.where(noisy_x[:,idxs[yi]]>intersection)]
                #concept drift
                ys = ys[np.where(noisy_x[:,idxs[yi]]>intersection)]
            if plot is True:                                                            
                plt = plot_regression(model,noisy_x,ys,idxs)
            else:
                plt = None
            if fig != None:                                                            
                plt = plot_regression(model,noisy_x,ys,idxs)
                plt.savefig(fig)
            try:                                                            
                plt
            except Exception as e:                                                            
                plt = None
        except Exception as e:
            print('Bad explanation parameters given')
    model.kernel1 = truek
    #add code for effect size or feature transformation    
    #given expected criteria, return accuracies per instance
    try:    	
        return pos_explanation_scores, neg_explanation_scores, plt
    except Exception as e:
        return pos_explanation_scores, neg_explanation_scores, None  
