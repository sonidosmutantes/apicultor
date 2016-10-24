from smst.utils.math import to_db_magnitudes
import numpy as np
from essentia.standard import *
import os
import sys

Usage = "./mute_removal.py [DATA_PATH]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        DATA_PATH = sys.argv[1] 

    	if not os.path.exists(DATA_PATH):                         
		raise IOError("Must download sounds")

	for subdir, dirs, files in os.walk(DATA_PATH):
	    for f in files:
		    print( "Rewriting without silence in %s"%f )
		    audio = MonoLoader(filename = DATA_PATH+'/'+f)() 
		    db_mag = to_db_magnitudes(audio) 
		    silence_threshold = -130 #complete silence
		    loud_audio = np.delete(audio, np.where(db_mag <  silence_threshold))#remove it
		    os.remove(subdir+'/'+f)
		    MonoWriter(filename = subdir+'/'+os.path.splitext(f)[0]+'.ogg', format = 'ogg')(loud_audio)                         

    except Exception, e:
        print(e)
        exit(1)
