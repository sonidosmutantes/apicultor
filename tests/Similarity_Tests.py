import os
import sys
import unittest
import numpy as np

# sys.path.append(os.path.dirname(os.path.abspath(''))) use this if you've compiled apicultor

# from apicultor.SoundSimilarity import * # import from compiled apicultor 

from SoundSimilarity import * # first copy SoundSimilarity to tests directory

files_dir = '/full/path/to/apicultor/apicultor/data/bajo' # full path of sounds tag
files = get_files(files_dir)
dics = get_dics(files_dir)

class Testfiles(unittest.TestCase):

    def test_get_files(self):
        self.assertNotEqual(plot_similarity_clusters(files,dics), -1) # make sure clustering doesn't fail
 
if __name__ == '__main__':
    unittest.main()
