#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np

sigmoid = lambda x: 1 / (1 + np.e ** (-1 * x))
g = lambda lab, a0, Q: 1 - lab * np.sum(a0 * Q)                 
Q_a = lambda a, Q: np.sum(a) - (np.sum(np.sum(a * Q, axis = 0), axis = 0)/2)
def attention(Q, K, V):
    num = np.array(Q.dot(K.T))
    denum = np.sqrt(K.shape[0])
    return sigmoid(num / denum) * V
