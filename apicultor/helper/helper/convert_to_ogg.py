#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

Usage = "./convert_to_ogg.py [DATA_PATH]"

def main():
    if len(sys.argv) < 2:
        print("\nBad amount of input arguments\n", Usage, "\n")
        sys.exit(1)

    try:                                                       
        DATA_PATH = sys.argv[1] 
        if not os.path.exists(DATA_PATH):                         
            raise IOError("Must download sounds")                              
        os.mkdir(DATA_PATH+'/duration')                       
        for subdir, dirs, files in os.walk(DATA_PATH):        
            for f in files:                                   
                print(( "Processing %s"%f ))          
                subprocess.call(['ffmpeg', '-i', subdir + '/' + f, subdir + '/' + f.split('.')[0] + '.wav']) 
                subprocess.call(['ffmpeg', '-i', subdir + '/' + f.split('.')[0] + '.wav', subdir + '/' + f.split('.')[0] + '.ogg']) 
                subprocess.call(['rm', '-f', subdir + '/' + f])
                subprocess.call(['rm', '-f', subdir + '/' + f.split('.')[0] + '.wav'])                                           
    except Exception:                                                                                                          
       sys.exit(1) 

if __name__ == '__main__': 
    main()
