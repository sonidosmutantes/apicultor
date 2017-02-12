#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np

g = lambda lab, a0, Q: 1 - lab * np.sum(a0 * Q)                 
Q_a = lambda a, Q: np.sum(a) - (np.sum(np.sum(a * Q, axis = 0), axis = 0)/2)
