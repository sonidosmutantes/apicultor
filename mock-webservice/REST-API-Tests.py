#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import unittest
import urllib2

URL_BASE = "http://127.0.0.1:5000"
# URL_BASE = "http://api.redpanal.org.ar"


class Test_REST_API(unittest.TestCase):

    def test_pista_audio(self):
        call = '/pistas/23/audio'
        response = urllib2.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("ID 23"), -1)

    def test_pista_descriptor(self):
        call = '/pistas/76/descriptor'
        response = urllib2.urlopen(URL_BASE + call).read()
        # print(response)
        self.assertNotEqual(response.find("ID 76"), -1)
        self.assertNotEqual(response.find("json"), -1)

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
