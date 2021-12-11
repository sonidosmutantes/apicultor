from .subproblem import s, sigmoid, attention
import numpy as np

def attention_sga(x,y,a=None):
    if a == None:
        a = np.zeros(x.shape[1])
    else:
        a = np.resize(a,x.shape[1])        
    for i in range(5):
        for tau in range(x.shape[1]):
            gh = abs(1-np.min(attention(np.mat(x),np.mat(y),a[tau]))) * np.gradient(x.T*a[tau],axis=0)[tau]
            s = lambda smax, b, B: 1/smax+(smax-(1/smax))*((b-1)/(B-1))
            et = sigmoid(np.array(gh))
            s_max = 1.5
            s1 = s(1.5,tau,x.shape[1])    
            ql =(s_max * np.abs(np.cosh(s1 * et)+1)) / (s1*np.abs(np.cosh(s1 * et)+1))
            a[tau] += ql
    return a
