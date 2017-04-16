import json
import sys
import urllib2

from imirdbapi import * 

class FreesoundAPI(IMirDbApi):
    __api_key = ""

    def setApiKey(self, api_key):
        self.__api_key = api_key

    def search_by_id(self, id=""):
        """
            Returns a json
        """
        request = urllib2.Request("http://www.freesound.org/apiv2/sounds/%s/"%id)
        request.add_header('Authorization','Token %s'%self.__api_key)
        response = urllib2.urlopen(request).read()
        return json.JSONDecoder().decode(response)
    #search_by_id()
    
    def search_by_content(self, content=""):
        """
            Returns a json
        """
        content = content.replace (" ","%20")
        #TODO: use urllib3.request.urlencode.
        request = urllib2.Request("http://www.freesound.org/apiv2/search/content/?target=%s"%content)
        request.add_header('Authorization','Token %s'%self.__api_key)
        response = urllib2.urlopen(request).read()
        return json.JSONDecoder().decode(response) 
        #search_by_content()

    #TODO: add as a parameter and callback function processing in other methods
    def json_to_id_list(self, json_content):
        """
            Input: json
            Output: id list
        """
        ids = list()
        # print json_content
        for r in json_content["results"]:
            ids.append(r["id"])
            # print r
        return ids
    #json_to_id_list()

if __name__ == '__main__':
    # JSON config file
    config = ""
    try:
        config = json.load( open(".apicultor_config.json",'r') )
    except Exception, e:
        print(e)
        print("No json config file or error.")
        sys.exit(2)

    api = FreesoundAPI()
    api.setApiKey( config["Freesound.org"][0]["API_KEY"] )
    #api.search_by_id(31362)
    
    #content = "lowlevel.pitch.mean:220"
    #content = "lowlevel.pitch.mean:220 AND lowlevel.pitch.var:0"
    content = "lowlevel.spectral_centroid.mean:100"
    
    # TODO: ver  "distance_to_target": 5.960464477539063e-08

    result = api.search_by_content(content)
    print( api.json_to_id_list(result) )