import os
import urllib2
from bs4 import BeautifulSoup
import re
import wget

website = 'http://redpanal.org'

host = 'http://redpanal.org/tag/'

tag = 'teclado' 

search = tag+'/audios' 


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

      return  list(set([website+'/media'+audiopath for audiopath in audiopaths]))

def download_files():

    paths_in_website = searchfiles(website, host, tag, search)

    [wget.download(path_in_website, out = tag) for path_in_website in paths_in_website]


download_files()





