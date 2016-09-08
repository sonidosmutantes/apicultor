#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import unittest
import urllib2

URL_BASE = "http://127.0.0.1:5000"  #TODO: get from a config file
# URL_BASE = "http://api.redpanal.org.ar"


class Test_REST_API(unittest.TestCase):

    def test_list_pistas(self):
        call = '/list/pistas'
        response = urllib2.urlopen(URL_BASE + call).read()
        # for file in response.split('\n'):
        #     print(file)
        self.assertNotEqual(response.find("1288.ogg"), -1)
        self.assertNotEqual(response.find("795.ogg"), -1)

    def test_mir_samples_hfc_greater(self):
        call = '/search/mir/samples/HFC/greaterthan/40000/5' #max 5 results
        response = urllib2.urlopen(URL_BASE + call).read()
        # for file in response.split('\n'):
        #     print(file)
        self.assertNotEqual(response.find("984_sample4.json"), -1)
        self.assertNotEqual(response.find("984_sample1.json"), -1)

    def test_mir_samples_hfc_less(self):
        call = '/search/mir/samples/HFC/lessthan/1000/5' #max 5 results
        response = urllib2.urlopen(URL_BASE + call).read()
        # for file in response.split('\n'):
        #     print(file)
# ./samples/982_sample3.json
# ./samples/1288_sample3.json
# ./samples/982_sample0.json
# ./samples/1288_sample0.json
        self.assertNotEqual(response.find("982_sample3.json"), -1)
        self.assertNotEqual(response.find("1288_sample3.json"), -1)

    def test_list_samples(self):
        call = '/list/samples'
        response = urllib2.urlopen(URL_BASE + call).read()
        # for file in response.split('\n'):
        #     print(file)
        # self.assertNotEqual(response.find(".wav"), -1)

    def test_pista_audio(self):
        call = '/pistas/126/audio'
        response = urllib2.urlopen(URL_BASE + call).read()
        #print(response)
        self.assertNotEqual(response.find("126"), -1)

    def test_pista_descriptor(self):
        #INFO: descriptor not in the repo, run mir analysis first to generate json file (WARNING)
        call = '/pistas/76/descriptor' #id 76 (no existente en la DB) retorna 404
        try:
            response = urllib2.urlopen(URL_BASE + call).read()
        except Exception, e:
            self.assertNotEqual(str(e).find("Error 404"), -1)

        call = '/pistas/126/descriptor' # id 126 (existente, retorna json)
        response = urllib2.urlopen(URL_BASE + call).read()
        self.assertNotEqual(response.find("lowlevel.dissonance.mean"), -1)
        print("FIXME: assert value. Return float not integer")

    def test_pista_search(self):
        call = '/search/bass/10'
        response = urllib2.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("10 resultados"), -1)

    def test_last_5_searchs(self):
        call = '/search/last/5'
        response = urllib2.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("últimas 5"), -1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
