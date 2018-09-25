#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

fields_list = [ 
  "lowlevel.dissonance.mean",
  "lowlevel.mfcc_bands.mean",
  "sfx.inharmonicity.mean",
  "rhythm.bpm.mean",
  "lowlevel.spectral_contrast.mean",
  "lowlevel.spectral_centroid.mean",
  "rhythm.bpm_ticks.mean",
  "lowlevel.mfcc.mean",
  "loudness.level.mean",
  "metadata.duration.mean",
  "lowlevel.spectral_valleys.mean",
  "lowlevel.hfc.mean"]

Usage = "./Build_csv.py [FILES_DIR] [FILE.csv]"

def main():
    if len(sys.argv) < 3:
        print(("\nBad amount of input arguments\n", Usage, "\n" ))
        sys.exit(1)

    csv_filename = sys.argv[2]
    fcsv = open(csv_filename,'w')
    fields = ""
    for i in fields_list:
        fields += i + ","
    fcsv.write("name,"+fields+"\n")

    try:
        files_dir = sys.argv[1]

        if not os.path.exists(files_dir):                         
            raise IOError("Must download sounds")

        for subdir, dirs, files in os.walk(files_dir):
            for f in files:
                base, ext = os.path.splitext(f)
                if os.path.splitext(f)[1]==".json":
                    data = json.load( open(files_dir + "/" + f,'r') )

                    try:
                        line = str(base)+","
                        for i in fields_list:
                            line += str( data[i] )+","
                        #print(line)
                        fcsv.write(line+"\n")
                    except Exception as e:
                        print(( "Error with %s: %s"%(f,str(e))))
    except Exception as e:
        print(e)
        exit(1)
    #finally:
    #    conn.close()
    fcsv.close()

if __name__ == '__main__': 
    main()

