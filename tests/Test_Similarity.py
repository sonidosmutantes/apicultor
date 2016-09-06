#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import os
import sys
import unittest
import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(''))) use this if you've compiled apicultor

# from apicultor.SoundSimilarity import * # import from compiled apicultor 

#sys.path.append(os.path.dirname(os.path.abspath('..'))) #use this if you've compiled apicultor
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..'))
from SoundSimilarity import * # first copy SoundSimilarity to tests directory

files_dir = '/full/path/to/apicultor/apicultor/data/bajo' # full path of sounds tag
files = get_files(files_dir)
dics = get_dics(files_dir)

class Testfiles(unittest.TestCase):

    def test_get_files(self):
	#TODO: assert expected results
	print("TODO: compare with expected results (add assertions)")
        self.assertNotEqual(plot_similarity_clusters(files,dics), -1) # make sure clustering doesn't fail
 
if __name__ == '__main__':
    unittest.main()
