import numpy as np

def BTC(y, yhat):
    """
    Backward trust compatibility measures target dependency in points after updating their targets                                                           
    :param y: targets                                                                                         
    :param yhat: predictions        
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

def BEC(y, yhat,keep_index=None,return_array=None):
    """
    Backward error compatibility measures relativity error between points updating their targets                                                           
    :param y: classes                                                                                         
    :param yhat: predictions
    :param keep_index: bool indicating index storage or list indicating value updating                                                                                         
    :param return_array: whether to return or not conflict lists
    :returns:                                                                                                         
      - hat becs: target-wise error compatibility         
      - bec score: target-wise misclassified data indexes  
    """
    hat_becs = []
    istarget_prev = 1e-4
    istarget_con = 1e-4    
    hat_cons = []    
    for i in np.unique(y):
        if keep_index== None:
            iscon = []
        elif keep_index== True:
            iscon = []
        else:
            iscon = [0 for i in range(len(y))]
            try:
                iscon[keep_index[i]] = 1
            except Exception as e:
                pass
        for j in range(len(y)):
            if y[j] != i:
                if yhat[j] != i:
                    istarget_prev += 1
                    istarget_con += 1
                    if keep_index == None:
                        iscon.append(1)
                    elif keep_index == True:
                        iscon.append(1)
                    else:
                        iscon[j]=1
            elif yhat[j] != i:
                istarget_con += 1
                if keep_index == None:
                    iscon.append(1)            
                elif keep_index == True:
                    iscon.append(1)  
                else:
                    iscon[j] = 1  
            else:
                if keep_index == None:
                    iscon.append(0)            
                elif keep_index == True:
                    iscon.append(0)
                else:
                    iscon[j] = 0
        hat_becs.append(1-(istarget_prev / istarget_con))
        iscon = np.array(iscon)
        hat_cons.append(np.where(iscon == 1))
        istarget_prev = 1e-4
        istarget_con = 1e-4        
    return hat_becs, hat_cons
