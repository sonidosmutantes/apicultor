#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from .subproblem import sigmoid, attention,s
import numpy as np

#taken from Pegasos algorithm by avaitla
def SGD(a, lab, Q, lr):
    for i in range(20):
        iterations = 1
        for tau in range(len(a)):
            if a[tau] > 0:
                wx = a @ Q[:,tau]
                a[tau] *= (1 - 1/iterations)
                if(lab[tau]*wx < 1):
                    a[tau] += lab[tau]/(lr * iterations)
                iterations += 1
    return a

def attention_sgd(x,y,a=None):
    if a == None:
        a = np.zeros(x.shape[1])
    else:
        a = np.resize(a,x.shape[1])        
    for i in range(5):
        for tau in range(x.shape[1]):
            gh = abs(1-np.min(attention(np.mat(x),np.mat(y),a[tau]))) * np.gradient(x.T*a[tau],axis=0)[tau]
            et = sigmoid(np.array(gh))
            s_max = 1.5
            s1 = s(1.5,tau,x.shape[1])    
            ql =(s_max * np.abs(np.cosh(s1 * et)+1)) / (s1*np.abs(np.cosh(s1 * et)+1))
            a[tau] -= ql
        cross_entropy = -np.sum(np.log((np.mat(x).dot(a))+1e-9)*y)/len(x)
        if 1e-9 > cross_entropy:
            break
            return a                          
    return a
