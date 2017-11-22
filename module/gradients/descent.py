#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot as ssd

#taken from Pegasos algorithm by avaitla
def SGD(a, lab, Q, reg_param):
    iterations = 1
    for i in xrange(7):
        for tau in xrange(len(a)):
            wx = ssd(a, Q[tau,:], dense_output = True)
            a = a * (1-1/iterations)
            if(lab[tau]*wx < 1):
                a[tau] = lab[tau]/(reg_param * iterations)
            iterations += 1
    return a
