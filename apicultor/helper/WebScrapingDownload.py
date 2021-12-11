#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import os
import sys
import urllib.request, urllib.error, urllib.parse
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
    resp = urllib.request.urlopen(tags_url+search+page)
    resp_pages = []
    while index < 30: #look for 30 pages
        index += 1
        resp_pages.append(urllib.request.urlopen(resp.geturl()+str(index)))
    htmlcodes = [BeautifulSoup(i) for i in resp_pages]

    links = []

    for htmlcode in set(htmlcodes):
        for link in htmlcode.findAll('a', attrs={'href': re.compile("^/a/")}):
            links.append(link.get('href'))

    urls = [host+link for link in links]

    openurl = [urllib.request.urlopen(url) for url in set(urls)]

    htmlcodes = [BeautifulSoup(openurl) for i, openurl in enumerate(openurl)]

    audiopaths = re.findall('<a href="/media?\'?([^"\'>]*)', str(htmlcodes))
      
    for i in audiopaths:
        if i.endswith('.jpg'):
            audiopaths.remove(i) #avoid downloading pictures

    return  list(set([host+'/media'+audiopath for audiopath in audiopaths]))

def download_files(tag, sounds_path):
    paths_in_website = searchfiles(tags_url, host, tag, search)

    [wget.download(path_in_website, out = sounds_path) for path_in_website in paths_in_website]


Usage = "./WebScrapingDownload.py [TAGNAME] [DATA_DIR]"
def main(): 
    if len(sys.argv) < 3:
        print("\nBad amount of input arguments\n", Usage, "\n")
        sys.exit(1)

    try:
        tagName = sys.argv[1]

        try:
            os.mkdir(sys.argv[2])
        except:
            pass

        search = tagName+'/audios' 
        print(( "Downloading most files with tag %s"%tagName ))
        download_files(tagName, sys.argv[2])

    except Exception as e:
        print(e)
        exit(1)

if __name__ == '__main__':
    main()
