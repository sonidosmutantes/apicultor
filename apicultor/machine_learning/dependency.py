import numpy as np

def BTC(y, yhat):
    """
    Backward trust compatibility measures target dependency in points after updating their targets                                                           
    :param y: scaled features                                                                                         
    :returns:                                                                                                         
      - btc score: target-wise trust compatibility         
    """
    hat_btcs = []
    istarget_prev = 1e-4
    istarget_dep = 1e-4    
    for i in np.unique(y):
        for j in range(len(y)):
            if y[j] == i:
                if yhat[j] == i:
                    istarget_prev += 1
                    istarget_dep += 1
            elif yhat[j] == i:
                istarget_dep += 1            
        hat_btcs.append(1-(istarget_prev / istarget_dep))
        istarget_prev = 1e-4
        istarget_dep = 1e-4
    return hat_btcs 

def BEC(y, yhat):
    """
    Backward error compatibility measures relativity error between points updating their targets                                                           
    :param y: scaled features                                                                                         
    :returns:                                                                                                         
      - bec score: target-wise error compatibility         
    """
    hat_becs = []
    istarget_prev = 1e-4
    istarget_con = 1e-4      
    for i in np.unique(y):
        for j in range(len(y)):
            if y[j] != i:
                if yhat[j] != i:
                    istarget_prev += 1
                    istarget_con += 1
            elif yhat[j] != i:
                istarget_con += 1            
        hat_becs.append(1-(istarget_prev / istarget_con))
        istarget_prev = 1e-4
        istarget_con = 1e-4        
    return hat_becs    
