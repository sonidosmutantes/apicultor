#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import os
import sys
import unittest
import numpy as np

# from apicultor.SoundSimilarity import * # import from compiled apicultor 
sys.path.append(os.path.join(os.path.dirname(os.path.realpath('__file__')), '..'))
from ..machine_learning.SoundSimilarity import * # first copy SoundSimilarity to tests directory


#Example
files_dir = '../data/bajo' # full path of sounds tag
files = get_files(files_dir)
dics = get_dics(files_dir)
descriptors = desc_pair(files,dics).descriptors
files_features = desc_pair(files,dics).files_features
keys = desc_pair(files,dics).keys
desc1, desc2, in1, in2 = get_desc_pair(descriptors, files_features, keys)

class Testfiles(unittest.TestCase):

    def test_get_files(self):
        self.assertIsNotNone(np.any(plot_similarity_clusters(desc1, desc2))) # assert there are clusters
    def test_clusters(self):
        self.assertEqual(np.any(plot_similarity_clusters(desc1, desc2)) != -1,True) # assert there are labels
        self.assertEqual(np.any(np.where(plot_similarity_clusters(desc1, desc2)==1)),True)
 
if __name__ == '__main__':
    unittest.main()
