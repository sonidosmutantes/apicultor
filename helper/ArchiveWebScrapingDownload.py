#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import urllib2
from bs4 import BeautifulSoup
import re
import wget

"""
    Descarga hasta 310 sonidos del archivo redpanalero tomando
    en cuenta el tag
    Los guarda en una carpeta con el mismo nombre (del tag)
"""

def searchfiles():
    error_count = 0
    resp = [] 
    project = 0
    while project <= 1013:
        try:
            resp.append(urllib2.urlopen('http://archivo.redpanal.org/project/'+str(project)))                         
        except Exception, e:
            print ("Project no. " + str(project) + " doesn't exist. Looking for more projects")    
            error_count += 1 #bypass Error 500
        project += 1

    htmlcodes = [BeautifulSoup(i) for i in resp]


    links = []

    for htmlcode in set(htmlcodes):
        for link in htmlcode.findAll('a', attrs={'href': re.compile('^/static/')}):
            links.append(link.get('href'))

    urls = ["http://archivo.redpanal.org" + link for link in links]

    for i in urls:
        if i.endswith('.jpg') or i.endswith('.zip') or i.endswith('.rar') or i.endswith('.doc'):
            audiopaths.remove(i) #avoid downloading non music formats

    return  list(set(urls))

def download_files():
    paths_in_website = searchfiles()

    [wget.download(path_in_website, out = "archive") for path_in_website in paths_in_website]


Usage = "./ArchiveWebScrapingDownload.py"
if __name__ == '__main__':
    try:

        try:
            os.mkdir("archive")
        except:
            pass

        download_files()

    except Exception, e:
        print(e)
        exit(1)
