#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import flask
from flask import jsonify
from flask_autodoc import Autodoc
import logging
import os
import json
from flask import abort
import itertools

#TODO: get from config files
DATA_PATH = "./data"
SAMPLES_PATH = "./samples"

app = flask.Flask(__name__)
auto = Autodoc(app)

"""
Allowed extensions by redpanal.org are: mp3, ogg, oga, flac
.wav is not included, but useful for internal representations
"""
ext_filter = ['.mp3','.ogg','.ogg','.wav']

#########################
# Helper functions
#########################

#TODO: definir una interfaz, luego implementar (leyendo json files y accediendo la BD)
#from JsonMirFilesData import *
import mir.db.JsonMirFilesData as mirdata
#TODO: see MIR_State_Machine classes
#from DBMirData import *

#########################
# API functions
#########################

@app.route('/documentation')
def documentation():
    """
        Doc generation
    """
    return auto.html('public').replace("&amp;lt;br&amp;gt;","") # quick fix with replace (html gen)


@auto.doc('public')
@app.route('/pistas/<int:id>/audio', methods=['GET'])
def get_pista_audio(id):
    """
        Retorna el full path donde esta descargado el audio con ese id
    """
    return mirdata.get_url_audio(id)
    # desc = json.load( open(DATA_PATH + "/" + str(id) + ".json",'r') )
    #return flask.send_file( DATA_PATH + "/" + desc['filename'] )



@auto.doc('public')
@app.route('/pistas/<int:id>/descriptor', methods=['GET'])
def get_pista_descriptor_file(id):
    """
        json con descriptores
    """
    try:
        with open(DATA_PATH + "/" + str(id) + ".json",'r') as f:
            return( f.read() )
    except:
        abort(404) #not found


@auto.doc('public')
@app.route('/pistas/<int:id>/descriptor/<descname>', methods=['GET'])
def get_pista_descriptor_value(id, descname):
    """
        Path con el archivo  json con descriptores
    """
    desc = json.load( open(DATA_PATH + "/" + str(id) + ".json",'r') )
    output = "not found"
    try:
        output = str( desc[descname][0] )
    except Exception, e:
        app.logger.error( e )
        abort(404) #not found
    return output

@auto.doc('public')
@app.route('/pistas/<int:id>', methods=['GET'])
def get_pista(id):
    """
        Info de la pista. Json con path a archivos de audio y del descriptor
    """
    return jsonify(id=id,
                   desc=os.path.abspath(DATA_PATH) + "/" + str(id) + ".json",
                   audio=mirdata.get_url_audio(id))


@auto.doc('public')
@app.route('/search/<query>/<int:maxnumber>', methods=['GET'])
def get_search_query(query, maxnumber):
    """
        Search result of query
    """
    app.logger.warning("Falta implementar")
    return ("Json con %s resultados, cada uno con id y audio+desc url,correspondientes con %s" % (maxnumber, query))

###################################################################################
### MIR ###

@auto.doc('public')
@app.route('/search/mir/samples/samecluster/<samplename>/<int:maxnumber>', methods=['GET'])
def get_search_mir_samecluster(samplename, maxnumber):
    """
        Returns a list of samples in the same cluster that the input sample name
    """
    outlist = mirdata.get_list_of_sample_files_same_cluster(samplename)
    top5 = itertools.islice(outlist, maxnumber)
    # TODO: Falta implmementar el formato json, por ahora es una lista!
    #       o dar como opción PLAIN/JSON, en plano es más cómodo para laburar en SuperCollider?
    output = ""
    for f in top5:
        output += f + "\n"
    return(output)

@auto.doc('public')
@app.route('/search/mir/samples/<querydescriptor>/greaterthan/<int:fixedfloatvalue>/<int:maxnumber>', methods=['GET'])
def get_search_mir_query_greater(querydescriptor, fixedfloatvalue, maxnumber):
    """
        Search result of query (mayor)
        Falta implmementar el formato json, por ahora es una lista!
        JSON con %i pistas con el parxmetro %s mayor que %f" % (maxnumber,querydescriptor,comp_value)
    """
    outlist = mirdata.get_list_of_files_comparing(SAMPLES_PATH, querydescriptor, fixedfloatvalue, ">")
    top5 = itertools.islice(outlist, maxnumber)
    # TODO: Falta implmementar el formato json, por ahora es una lista!
    #       o dar como opción PLAIN/JSON, en plano es más cómodo para laburar en SuperCollider?
    output = ""
    for f in top5:
        output += f + "\n"
    return(output)


@auto.doc('public')
@app.route('/search/mir/samples/<querydescriptor>/lessthan/<int:fixedfloatvalue>/<int:maxnumber>', methods=['GET'])
def get_search_mir_query_less(querydescriptor, fixedfloatvalue, maxnumber):
    """
        Search result of query (menor)
        Falta implmementar el formato json, por ahora es una lista!
        JSON con %i pistas con el parxmetro %s mayor que %f" % (maxnumber,querydescriptor,comp_value)
    """
    outlist = mirdata.get_list_of_files_comparing(SAMPLES_PATH, querydescriptor, fixedfloatvalue, "<")
    top5 = itertools.islice(outlist, maxnumber)
    # TODO: Falta implmementar el formato json, por ahora es una lista!
    #       o dar como opción PLAIN/JSON, en plano es más cómodo para laburar en SuperCollider?
    output = ""
    for f in top5:
        output += f + "\n"
    print "output"
    print output
    return(output)


@auto.doc('public')
@app.route('/search/last/<int:number>', methods=['GET'])
def get_last_searchs(number):
    app.logger.warning("Falta implementar")
    return ("Json con las últimas %i búsquedas" % (int(number)))


@auto.doc('public')
@app.route('/search/tag/<tag1>', methods=['GET'])
def get_tag_search(tag1):
    """
        Search by tag name
    """
    app.logger.warning("Falta implementar")
    return ("Resultado de buscar por tag %s" % (tag1))


@auto.doc('public')
@app.route('/list/pistas', methods=['GET'])
def list_pistas():
    """
        list audio files (DATA)
    """
    app.logger.warning("Falta implementar en formato definitivo") #todas? o poner un máximo?
    return( mirdata.get_list_of_files(DATA_PATH) )

@auto.doc('public')
@app.route('/list/samples', methods=['GET'])
def list_samples():
    """
        list sample files (segmented pistas)
    """
    app.logger.warning("Falta implementar en formato definitivo") #todos? o poner un máximo?
    return( mirdata.get_list_of_files(SAMPLES_PATH) )

if __name__ == "__main__":
    file_handler = logging.FileHandler('mock_redpanal_api_ws.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    #app.run()
    app.run( debug=True, host="0.0.0.0", port=5000 )

