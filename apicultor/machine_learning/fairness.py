import numpy as np

def p_rule(y_predicted,y,theta,x,proba): 
    if -1e-4 < 1/y.size*np.sum((y-y) * (theta @ x.T)) < 1e-4: 
        return min(-np.sum(np.log(proba),axis=1))
    else: 
        return False       

def unprotection_score(old_loss,fx,y):
    new_loss = np.mean((y-np.sign(fx))**2)
    
    unprotected = ((1.0 + 1) * old_loss) - new_loss
    print('Unprotection score: '+str(unprotected))

#both measure fairness for regression problems
def ind_fairness(fxi,fxj,y):
    return 1/(len(fxi)**2) * (np.sum(y.size*((fxi-fxj)**2)))

def group_fairness(fairness):
    return fairness**2
