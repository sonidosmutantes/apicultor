#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np

def dsvm_low_a(a):
    a[a < 0] = 0.0
    return a

def dsvm_high_a(a, cw, c):
    a = np.array([min(a[i], c * cw[i] * cw[i]) if a[i] > 0 else a[i] for i in range(len(a))])
    return a

def es(a, lab, features):
    if any(a > 0.0):                                               
        return np.median((lab[a > 0.0] - np.sum(a[a > 0.0] * lab[a > 0.0] * features[a > 0.0].T).T * features[a > 0.0].T).T)
    else:
        return 0
