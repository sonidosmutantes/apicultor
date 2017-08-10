#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import unittest
import sys
import json

# from mir.db.api import MirDbApi
# from mir.db import FreesoundDB
from mir.db.FreesoundDB import FreesoundDB

class Test_Freesound_API(unittest.TestCase):
    
    def setUp(self):
        # Load JSON config file
        config = ""
        try:
            config = json.load( open(".apicultor_config.json",'r') )
        except Exception, e:
            print(e)
            print("No json config file or error.")
            sys.exit(2)

        self.api = FreesoundDB()
        self.api.set_api_key( config["Freesound.org"][0]["API_KEY"] )
    #set_up()

    def test_search_by_id(self):
       sound = self.api.search_by_id(31362)
       self.assertEqual( sound.name, 'coming soon.wav')
       self.assertEqual( sound.url, 'https://www.freesound.org/people/Corsica_S/sounds/96541/')
# ('Description:', u'For the user "Stealth Inc.", my girlfriend saying "Coming soon". Recorded with an AT2020 microphone into an Apogee
# Duet and edited with Record.')
# ('Tags:', u'request female girl talk vocal voice american woman english speak')

    def test_mir_search(self):
        state = ""
        # sound_list = self.api.search_by_mir(state)

if __name__ == '__main__':
       #unittest.main()
    unittest.main(verbosity=2)