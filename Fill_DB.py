#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import MySQLdb
import json

fields_list = [ "`name`",
  "`lowlevel.dissonance.mean`",
  "`lowlevel.mfcc_bands.mean`",
  "`sfx.inharmonicity.mean` ",
  "`rhythm.bpm.mean` ",
  "`lowlevel.spectral_contrast.mean` ",
  "`lowlevel.spectral_centroid.mean` ",
  "`rhythm.bpm_ticks.mean` ",
  "`lowlevel.mfcc.mean` ",
  "`loudness.level.mean` ",
  "`metadata.duration.mean` ",
  "`lowlevel.spectral_valleys.mean`",
  "`lowlevel.hfc.mean`"]

Usage = "./Fill_DB.py [FILES_DIR]"
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)

    fields = ""
    for i in fields_list:
        fields += i + ","

    # values_list = list()
    # for i in range(len(fields_list)-1):
    #     values_list.append(1.3)
    # values = ""
    # for i in values_list:
    #     values += str(i) + ","

    # statement = "INSERT INTO sample (%s)  VALUES (\"%s\", %s);"%(fields[:-1],"sample.json",values[:-1])
    #print statement
    conn = MySQLdb.connect(host= "localhost",
                  user="hordia", #username to access created database
                  passwd="admin", #password to access created database
                  db="mir") #default database name
    x = conn.cursor()


    try:
        files_dir = sys.argv[1]

        if not os.path.exists(files_dir):                         
            raise IOError("Must download sounds")

# fields_list = [ "`name`",
#   "`lowlevel.dissonance.mean`",
#   "`lowlevel.mfcc_bands.mean`",
#   "`sfx.inharmonicity.mean` ",
#   "`rhythm.bpm.mean` ",

#   "`lowlevel.spectral_contrast.mean` ",

#   "`lowlevel.spectral_centroid.mean` ",
#   "`rhythm.bpm_ticks.mean` ",

#   "`lowlevel.mfcc.mean` ",
#   "`loudness.level.mean` ",
#   "`metadata.duration.mean` ",
#   "`lowlevel.spectral_valleys.mean`",
#   "`lowlevel.hfc.mean`"]
        for subdir, dirs, files in os.walk(files_dir):
            for f in files:
                if os.path.splitext(f)[1]==".json":
                    data = json.load( open(files_dir + "/" + f,'r') )

                    try:
                        statement = "INSERT INTO sample (%s)  VALUES (\"%s\", %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f);"%(fields[:-1],f,float(data["lowlevel.dissonance.mean"]),float(data["lowlevel.mfcc_bands.mean"]),float(data["sfx.inharmonicity.mean"]),float(data["rhythm.bpm.mean"]),float(data["lowlevel.spectral_contrast.mean"]),float(data["lowlevel.spectral_centroid.mean"]),float(data["rhythm.bpm_ticks.mean"]),float(data["lowlevel.mfcc.mean"]),float(data["loudness.level.mean"]),float(data["metadata.duration.mean"]),float(data["lowlevel.spectral_valleys.mean"]),float(data["lowlevel.hfc.mean"]) )
                        #print(statement)
                        try:
                           x.execute(statement)
                           conn.commit()
                        except:
                           conn.rollback()
                    except Exception, e:
                        print( "Error with %s: %s"%(f,str(e)))
    except Exception, e:
        print(e)
        exit(1)
    finally:
        conn.close()
