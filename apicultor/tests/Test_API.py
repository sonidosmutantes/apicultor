#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import unittest
import urllib.request, urllib.error, urllib.parse

URL_BASE = "http://127.0.0.1:5000"  #TODO: get from a config file
# URL_BASE = "http://api.redpanal.org.ar"

def count_response_lines(response):
    count = 0
    for line in response.split('\n'):
        count += 1
    return count

class Test_REST_API(unittest.TestCase):

    def test_list_pistas(self):
        """
            Lista pistas DB
        """
        call = '/list/pistas'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # for file in response.split('\n'):
        #     print(file)
        self.assertNotEqual(response.find("1288.ogg"), -1)
        self.assertNotEqual(response.find("795.ogg"), -1)

    def test_mir_samples_hfc_greater(self):
        """ HFC > 40. """
        call = '/search/mir/samples/HFC/greaterthan/40000/5' #max 5 results
        response = urllib.request.urlopen(URL_BASE + call).read()
        self.assertNotEqual(response.find("984_sample4.wav"), -1)
        self.assertNotEqual(response.find("984_sample1.wav"), -1)

    def test_mir_samples_hfc_less(self):
        """ HFC < 1. """
        call = '/search/mir/samples/HFC/lessthan/1000/5' #max 5 results
        response = urllib.request.urlopen(URL_BASE + call).read()
        self.assertNotEqual(response.find("982_sample3.wav"), -1)
        # self.assertNotEqual(response.find("1288_sample3.wav"), -1)

    def test_mir_samples_duration_greater(self):
        """ Duration > 2 seg """
        call = '/search/mir/samples/duration/greaterthan/2000/11' #max 10 results
        response = urllib.request.urlopen(URL_BASE + call).read()
        self.assertNotEqual(response.find("982_sample3.wav"), -1)
        self.assertNotEqual(response.find("795_sample0.wav"), -1)
        self.assertNotEqual(response.find("126_sample1.wav"), -1)
        self.assertNotEqual(response.find("983_sample3.wav"), -1)
        self.assertNotEqual(response.find("983_sample2.wav"), -1)
        self.assertNotEqual(response.find("984_sample4.wav"), -1)
        self.assertNotEqual(response.find("1288_sample3.wav"), -1)
        self.assertNotEqual(response.find("982_sample0.wav"), -1)
        self.assertNotEqual(response.find("982_sample3.wav"), -1)
        self.assertNotEqual(response.find("126_sample3.wav"), -1)

    def test_mir_samples_duration_less(self):
        """ Duration < 1 seg """
        call = '/search/mir/samples/duration/lessthan/1000/5' #max 5 results
        response = urllib.request.urlopen(URL_BASE + call).read()
        self.assertNotEqual(response.find("126_sample2.wav"), -1)
        # self.assertNotEqual(response.find("795_sample4.wav"), -1)
        # self.assertNotEqual(response.find("795_sample2.wav"), -1)

    def test_list_samples(self):
        """
            Lista samples DB
        """
        call = '/list/samples'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # for file in response.split('\n'):
        #     print(file)
        # self.assertNotEqual(response.find(".wav"), -1)

    def test_get_pista_info(self):
        """
            Getinfo de la pista (json)
        """
        call = '/pistas/126'
        response = urllib.request.urlopen(URL_BASE + call).read()
        #print(response)
        self.assertNotEqual(response.find("audio"), -1)
        self.assertNotEqual(response.find("desc"), -1)
        self.assertNotEqual(response.find("126"), -1)
    
    def test_get_pista_file_path_audio(self):
        """
            Get audio file path by ID
        """
        call = '/pistas/126/audio'
        response = urllib.request.urlopen(URL_BASE + call).read()
        #print(response)
        self.assertNotEqual(response.find("126"), -1)

    def test_pista_descriptor(self):
        """
            Get full json desc by id
        """
        #WARNING: descriptor not in the repo, run mir analysis first to generate json file (WARNING)
        call = '/pistas/76/descriptor' #id 76 (no existente en la DB) retorna 404
        try:
            response = urllib.request.urlopen(URL_BASE + call).read()
        except Exception as e:
            self.assertNotEqual(str(e).find("Error 404"), -1)

        call = '/pistas/126/descriptor' # id 126 (existente, retorna json)
        response = urllib.request.urlopen(URL_BASE + call).read()
        self.assertNotEqual(response.find("lowlevel.dissonance.mean"), -1)
        self.assertNotEqual(response.find("sfx.inharmonicity.mean"), -1)

    def test_pista_descriptor_value(self):
        """
            Get desc json value by id
        """
        #WARNING: descriptor not in the repo, run mir analysis first to generate json file (WARNING)
        call = '/pistas/126/descriptor/lowlevel.hfc.mean' # id 126 (existente, retorna json)
        response = urllib.request.urlopen(URL_BASE + call).read()
        self.assertEqual(int(response), 2)

    def test_pista_search(self):
        """
            Search query
        """
        call = '/search/bass/10'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("10 resultados"), -1)
        
        call = '/search/clarinete/10'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("10 resultados"), -1)

    def test_last_5_searchs(self):
        """
            Search last results
        """
        
        call = '/search/last/5'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("últimas 5"), -1)
    
    def test_search_by_tag(self):
        """
            Search by tag
        """
        call = '/search/tag/bajo'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("tag bajo"), -1)
    
    def test_search_mir_desc_greater_than(self):
        """
            Search by HFC >40
        """
        call = '/search/mir/samples/HFC/greaterthan/40000/5'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("984_sample4.wav"), -1)
        self.assertEqual( 5, count_response_lines(response) )
    
    def test_search_mir_desc_less_than(self):
        """
            Search by HFC <1 (5 results)
        """
        call = '/search/mir/samples/HFC/lessthan/1000/5'
        response = urllib.request.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("126_sample2.wav"), -1)
        self.assertEqual( 6, count_response_lines(response) )

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
