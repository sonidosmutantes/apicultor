import numpy as np

def validate_protection(decision):
    protected = np.all(decision >= 0)
    
    if protected == False:
        raise ValueError('Unprotected data, positive target decision should remain positive')

def unprotection_score(old_loss,fx,y):
    new_loss = np.mean((y-np.sign(fx))**2)
    
    unprotected = ((1.0 + 1) * old_loss) - new_loss
    print('Unprotection score: '+str(unprotected))

def p_rule(fx,y,y0,y1):
    #fx should be a normalized probability distribution
    non_prot_all = sum(fx == y1) # non-protected group is all values equal to the maximum probability 
    prot_all = sum(fx == y0) # protected group is all values with no probability 
    non_prot_pos = sum(fx[y == y1] == y1) # true positives
    prot_pos = sum(fx[y == y0] == y1) # false positives
    try: 
        frac_non_prot_pos = float(non_prot_pos) / float(non_prot_all) 
        frac_prot_pos = float(prot_pos) / float(prot_all)  
    except Exception as e: 
        return ValueError('No true positives in non protected group') 
    if frac_non_prot_pos == 0: 
          raise ValueError('P rule constraint is violated       as there is no fraction of non protected values belonging to a positive class') 
    p_rule = (frac_prot_pos / frac_non_prot_pos) * 100.0 
    if not min(frac_non_prot_pos,frac_prot_pos) >= p_rule/100: 
        return ValueError('P rule constraint is violated') 
    if .8 <= p_rule:
    	return p_rule
    else:
    	return ValueError("P Rule violated, should be at least 80%, current value is "+str(p_rule*100))
    return p_rule 


#both measure fairness for regression problems
def ind_fairness(fxi,fxj,y):
    return 1/(len(fxi)**2) * (np.sum(y[1].size*((fxi-fxj)**2)))

def group_fairness(fairness):
    return fairness**2

def dccp_constraint(x_sensitive,fx_sensitive):
    ziz = x_sensitive - np.mean(x_sensitive)
    p = np.sum(fx_sensitive.dot(ziz)) / len(x_sensitive)
    if not p >= 1e-4 and -1e-4 <= p:
        return ValueError('No protection for sensitive data')
    return p