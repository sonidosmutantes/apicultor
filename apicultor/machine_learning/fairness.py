import numpy as np

def p_rule(y_predicted,y,theta,x,proba): 
    """
    Unlike group fairness, satisfying p-rule should be enough
    to know if the model accomplishes statistical parity.
    If boundary error is independent of the product between dataset and regression,
    then statistical parity is satisfied and all data must be treated equally 
    """
    if -1e-4 < 1/(y.size*np.sum((y-y_predicted) * (theta @ x.T))) < 1e-4: 
        return min(-np.sum(np.log(proba),axis=1)) #tradeoff is inversely proportional to probability of being assigned more than 1 label
    else: 
        return False       

def unprotection_score(old_loss,fx,y):
    """
    A measure of conditional procedure accuracy equality (disparate mistreatment) between binary instances
    """
    new_loss = np.mean((y-np.sign(fx))**2)
    unprotected = ((1.0 + 1) * old_loss) - new_loss
    return unprotected

#both measure fairness for regression problems
def ind_fairness(fxi,fxj,y):
    """A measure of conditional parity, which is the local probability of being assigned to either targets
    of a binary problem
    """
    return 1/(len(fxi)**2) * (np.sum(y.size*((fxi-fxj)**2)))

def group_fairness(fairness):
    """Given a conditional parity between all binary problems, return global probability of being assigned to all targets
    """
    return fairness**2
