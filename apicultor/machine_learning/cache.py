#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from hashlib import sha512

# a memoize class for faster computing


class memoize:
    """
    The memoize class stores returned results into a cache so that those can be used later on if the same case happens
    """

    def __init__(self, func, size=96):
        """
        memoize class init
        :param func: the function that you're going to decorate
        :param size: your maximum cache size (in MB)
        """
        self.func = func
        self.known_keys = []  # a list of keys to save numpy arrays
        self.known_values = []  # a list of values to save numpy results
        self.size = int(size * 1e+6 if size else None)  # size in bytes
        self.size_copy = int(np.copy(self.size))

    def __call__(self, *args, **kwargs):
        key = sha512(
            bytes(str((args, kwargs, len(args[1]), len(args[2]))), 'utf-8')).hexdigest()
        if (not key in self.known_keys):  # when an ammount of arguments can't be identified from keys
            value = self.func(*args, **kwargs)  # compute function
            self.known_keys.append(key)  # add the key to your list of keys
            # add the value to your list of values
            self.known_values.append(value)
            if self.size:
                self.size -= value.__sizeof__()  # the assigned space has decreased
                # free cache when size of values goes beyond the size limit
                if (value.__sizeof__() > self.size):
                    del self.known_keys
                    del self.known_values
                    del self.size
                    self.known_keys = []
                    self.known_values = []
                    self.size = self.size_copy
                    self.known_keys.append(key)
                    self.known_values.append(value)
            return value
        else:  # if you've already computed everything
            # find your key and return your values
            i = self.known_keys.index(key)
            return self.known_values[i]
