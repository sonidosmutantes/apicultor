#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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
