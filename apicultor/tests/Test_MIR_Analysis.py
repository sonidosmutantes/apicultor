#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import unittest
import subprocess

class TestMIRAnalysis(unittest.TestCase):
    def test_error_count_in_data_dir(self):
        directory = "data"
        try:
            output = subprocess.check_output( "cd .. && ./run_mir_analysis.py %s"%directory, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            self.fail("Found errors processing 'data' directory. Error count: %i"%e.returncode)
            
    def test_error_count_in_samples_dir(self):
        directory = "samples"
        try:
            output = subprocess.check_output( "cd .. && ./run_mir_analysis.py %s"%directory, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            self.fail("Found errors processing 'samples' directory. Error count: %i"%e.returncode)

    #TODO: add test cases asserting expected results (values of well known sounds)
 
if __name__ == '__main__':
    unittest.main()
