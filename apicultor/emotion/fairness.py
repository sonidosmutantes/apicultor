import numpy as np

def validate_protection(decision):
    protected = np.all(decision >= 0)
    
    if protected == False:
        raise ValueError('Unprotected data, positive target decision should remain positive')

def unprotection_score(fx,y):
    old_loss=1
    new_loss = np.mean((y-np.sign(fx))**2)
    
    unprotected = ((1.0 + 1) * old_loss) - new_loss
    print('Unprotection score: '+str(unprotected))

def p_rule(fx,y):
    #fx should be a normalized probability distribution
    non_prot_all = sum(fx == 1) # non-protected group is all values equal to the maximum probability
    prot_all = sum(fx == 0) # protected group is all values with no probability
    non_prot_pos = sum(fx[y == 1] == 1.0) # non_protected in positive class
    prot_pos = sum(fx[y == 0.0] == 1.0) # protected in positive class
    frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all)
    frac_prot_pos = float(prot_pos) / float(prot_all) 
    if frac_non_prot_pos == 0:
          raise ValueError('P rule constraint is violated	as there is no fraction of non protected values	belonging to a positive class')
    p_rule = (frac_prot_pos / frac_non_prot_pos) * 100.0
    if not min(frac_non_prot_pos,frac_prot_pos) >= p_rule/100:
        raise ValueError('P rule constraint is violated')
    print('P Rule: ' + str(p_rule))  
    return p_rule      

#both measure fairness for regression problems
def ind_fairness(fxi,fxj):
    return 1/(len(fxi)**2) * np.sum(y[1].size*]((fxi-fxj)**2)))

def group_fairness(fairness):
    return fairness**2

def dccp_constraint(x,fx_sensitive):
    fx_sensitive = w*x_sensitive
    ziz = x_sensitive - np.mean(x_sensitive)
    p = np.sum(fx_sensitive * ziz) / len(x)
    if not p >= 1e-4 or -1e-4 <= p:
        raise ValueError('No protection for sensitive data')