#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import os
import sys
import urllib2
from bs4 import BeautifulSoup
import re
import wget

"""
    Descarga hasta 310 sonidos de la base de datos redpanalera tomando
    en cuenta el tag
    Los guarda en una carpeta con el mismo nombre (del tag)
"""

host = 'http://redpanal.org'

tags_url = host+'/tag/'

def searchfiles(tags_url, host, tag, search):
    page = '/?page=' 
    index = 0
    resp = urllib2.urlopen(tags_url+search+page)
    resp_pages = []
    while index < 30: #look for 30 pages
        index += 1
        resp_pages.append(urllib2.urlopen(resp.geturl()+str(index)))
    htmlcodes = [BeautifulSoup(i) for i in resp_pages]

    links = []

    for htmlcode in set(htmlcodes):
        for link in htmlcode.findAll('a', attrs={'href': re.compile("^/a/")}):
            links.append(link.get('href'))

    urls = [host+link for link in links]

    openurl = [urllib2.urlopen(url) for url in set(urls)]

    htmlcodes = [BeautifulSoup(openurl) for i, openurl in enumerate(openurl)]

    audiopaths = re.findall('<a href="/media?\'?([^"\'>]*)', str(htmlcodes))
      
    for i in audiopaths:
        if i.endswith('.jpg'):
            audiopaths.remove(i) #avoid downloading pictures

    return  list(set([host+'/media'+audiopath for audiopath in audiopaths]))

def download_files(tag):
    paths_in_website = searchfiles(tags_url, host, tag, search)

    [wget.download(path_in_website, out = tag) for path_in_website in paths_in_website]


Usage = "./WebScrapingDownload.py [TAGNAME]"
if __name__ == '__main__':
  
    if len(sys.argv) < 2:
        print "\nBad amount of input arguments\n", Usage, "\n"
        sys.exit(1)

    try:
        tagName = sys.argv[1]

        try:
            os.mkdir(tagName)
        except:
            pass

        search = tagName+'/audios' 
        print( "Downloading most files with tag %s"%tagName )
        download_files(tagName)

    except Exception, e:
        print(e)
        exit(1)
