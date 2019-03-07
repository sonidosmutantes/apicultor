#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np

s = lambda smax, b, B: 1/smax+(smax-(1/smax))*((b-1)/(B-1))
sigmoid = lambda x: 1 / (1 + np.e ** (-1 * x))
g = lambda lab, a0, Q: 1 - lab * np.sum(a0 * Q)                 
Q_a = lambda a, Q: np.sum(a) - (np.sum(np.sum(a * Q, axis = 0), axis = 0)/2)
def attention(Q, K, V):
    num = np.array(Q.dot(K.T))
    denum = np.sqrt(K.shape[0])
    return sigmoid(num / denum) * V
