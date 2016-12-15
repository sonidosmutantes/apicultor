#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

Usage = "./convert_to_ogg.py [DATA_PATH]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        DATA_PATH = sys.argv[1] 

    	if not os.path.exists(DATA_PATH):                         
		raise IOError("Must download sounds")

	from essentia.standard import *

	os.mkdir(DATA_PATH+'/duration')

	for subdir, dirs, files in os.walk(DATA_PATH):
	    for f in files:
		    print( "Processing %s"%f )
		    audio = MonoLoader(filename = DATA_PATH+'/'+f)() 
		    MonoWriter(filename = subdir+'/duration/'+os.path.splitext(f)[0]+'.ogg', format = 'ogg')(audio)                         

    except Exception, e:
        print(e)
        exit(1)
