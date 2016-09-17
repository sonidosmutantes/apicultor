#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import urllib2
from bs4 import BeautifulSoup
import re
import wget

"""
    Descarga los primeros diez archivos de la base de datos redpanalera tomando
    en cuenta el tag
    Los guarda en una carpeta con el mismo nombre (del tag)
"""

website = 'http://redpanal.org'

host = 'http://redpanal.org/tag/'

def searchfiles(website, host, tag, search):
    resp = urllib2.urlopen(host+search) 
    htmlcode = BeautifulSoup(resp)
    links = []

    for link in htmlcode.findAll('a', attrs={'href': re.compile("^/a/")}):
        links.append(link.get('href'))

    urls = [website+link for link in links]

    openurl = [urllib2.urlopen(url) for url in urls]

    htmlcodes = [BeautifulSoup(openurl) for i, openurl in enumerate(openurl)]

    audiopaths = re.findall('<a href="/media?\'?([^"\'>]*)', str(htmlcodes))
      
    while i in audiopaths:
        del audiopaths[i.endswith('.jpg')] #avoid downloading pictures

    return  list(set([website+'/media'+audiopath for audiopath in audiopaths]))

def download_files(tag):
    paths_in_website = searchfiles(website, host, tag, search)

    [wget.download(path_in_website, out = tag) for path_in_website in paths_in_website]


Usage = "./WebScrapingDownload.py [TAGNAME]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)


    try:
        tagName = sys.argv[1]

        #tagName = 'teclado' # Example 

        try:
            os.mkdir(tagName)
        except:
            pass

        search = tagName+'/audios' 
        print( "Downloading first 10 files with tag %s"%tagName )
        download_files(tagName)

    except Exception, e:
        print(e)
        exit(1)






