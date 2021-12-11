# -*- coding: UTF-8 -*-

import json
import sys
import urllib2
import subprocess
import random
import os

from mir.db.api import MirDbApi


DATA_PATH = "data"
SAMPLES_PATH = "samples"

# RedPanal API
class RedPanalDB(MirDbApi):
    __api_key = ""
    __url = ""

    def set_api_key(self, api_key):
        self.__api_key = api_key

    def __init__(self, url_base="http://127.0.0.1:5000"):
       self.__url = url_base

    # def search_by_id(self, id=""):
    #     return sound
    # #()

    # def search_by_content(self, content=""):

    def get_one_by_mir(self, mir_state):
        return self.search_by_mir(mir_state)

    def search_by_mir(self, mir_state):
        #FIXME generic call
        call = '/list/samples' #gets only wav files because SuperCollider
        response = urllib2.urlopen(self.__url + call).read()
        audioFiles = list()
        for file in response.split('\n'):
            if len(file)>0: #avoid null paths
                audioFiles.append(file)
                # print file
        
        sound_id = 0 #FIXME  (extract from filename)
        author = "Unknown" #FIXME (extract from metadata)
        for i in range(len(audioFiles)): #WARNING: file is chosen randomly
            file_chosen = audioFiles[ random.randint(0,len(audioFiles)-1) ]        
            if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000: #FIXME: prior remove 'silence' sounds from DB (ETL)
                return file_chosen, author, sound_id
    #()
