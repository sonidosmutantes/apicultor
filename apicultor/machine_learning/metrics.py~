import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix
from .visuals import plot_confusion_matrix

def getperfo(y,p,label_names,visual_title,plot=None,fig=None):
            """
            Given targets and preditions get the machine learning model performance metrics
            """
            cf = confusion_matrix(y, p)
            plt = plot_confusion_matrix(cf,label_names,visual_title)
            TP = np.diag(cf)
            FP = cf.sum(axis=0) - TP
            FN = cf.sum(axis=1) - TP
            TN = cf.sum() - (FP + FN + TP) 
            accuracy = (TP+TN)/(TP+FP+FN+TN) * 100
            recall = TP/(TP+FN) * 100 #precise data that does not belong to another class
            specificity = TN/(TN+FP) * 100 #data correctly classified as negative to other classes
            precision = TP/(TP+FP) * 100 #data correctly classified as positive to their classes
            harmonic_mean = 2 * ((precision * recall) / (precision + recall))
            if plot == True:          
                plt.show()          
            if fig != None:          
                plt.savefig(fig)
            return accuracy,recall,specificity,precision,harmonic_mean