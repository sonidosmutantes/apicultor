#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
import re
import wget

"""
    Descarga todos los sonidos del archivo redpanalero tomando
    en cuenta el tag
    Los guarda en una carpeta con el mismo nombre (del tag)
"""

def searchfiles():
    error_count = 0
    resp = [] 
    project = 0
    while project <= 1013:
        try:
            resp.append(urllib.request.urlopen('http://archivo.redpanal.org/project/'+str(project)))                         
        except Exception as e:
            print(("Project no. " + str(project) + " doesn't exist. Looking for more projects"))    
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

def download_files(output_path):
    paths_in_website = searchfiles()

    [wget.download(path_in_website, out = output_path) for path_in_website in paths_in_website]


Usage = "./ArchiveWebScrapingDownload.py [DATA_DIR]"
def main():
    if len(sys.argv) < 2:
        print("\nBad amount of input arguments\n", Usage, "\n")
        sys.exit(1)
    try:

        try:
            os.mkdir(sys.argv[1])
        except:
            pass

        download_files(sys.argv[1])

    except Exception as e:
        print(e)
        exit(1)

if __name__ == '__main__':
    main()
