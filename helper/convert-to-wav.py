#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess

#Requirements: ffmpeg

DATA_PATH = "."
ext_filter = ['.mp3','.ogg','.ogg']

for subdir, dirs, files in os.walk(DATA_PATH):
    for f in files:
        if os.path.splitext(f)[1] in ext_filter:
            print( "Processing %s"%f )
            # -y overwrites output file
            # subprocess.call("ffmpeg -i %s %s.wav"%(f,os.path.splitext(f)[0]), shell=True)
            subprocess.call("ffmpeg -i \"%s\" \"%s.wav\" -y"%(f,os.path.splitext(f)[0]), shell=True)
