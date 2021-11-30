import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes,title,cmap=plt.cm.Blues):
            """
            Given a confusion matrix and a list of classes, it is going
            to plot it
            """
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            thresh = cm.max() / 2.
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('y original')
            plt.xlabel('y predecida')    
            return plt
    
def plot_regression(model, x, y, idxs):
            """
            Uses model namespaces to visualize linear combination
            """
            plt.subplot(1, 2, 1)
            plt.scatter(x[:, idxs[0]], x[:, idxs[1]], c=y, s=30, cmap=plt.cm.Paired)
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
            K = np.array([xx.ravel(),yy.ravel()])
            for i in range(len(np.unique(len(x[0])))+1):
                K = np.vstack((K,[1 for i in yy.ravel()])) 
            K = K.T 
            return plt
