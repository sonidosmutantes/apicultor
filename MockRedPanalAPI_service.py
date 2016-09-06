#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import flask
from flask import jsonify
from flask_autodoc import Autodoc
import logging
import os
import json
from flask import abort

#TODO: get from config files
DATA_PATH = "./data"
SAMPLES_PATH = "./samples"

app = flask.Flask(__name__)
auto = Autodoc(app)

ext_filter = ['.mp3','.ogg','.ogg','.wav']

def get_url_audio(id):
    for subdir, dirs, files in os.walk(DATA_PATH):
        for f in files:
            if os.path.splitext(f)[1] in ext_filter and os.path.splitext(f)[0]==str(id):
                    return( os.path.abspath(DATA_PATH) + "/" + str(f) )
    abort(404) #not found


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
    return get_url_audio(id)



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
                   audio=get_url_audio(id))


@auto.doc('public')
@app.route('/search/<query>/<int:maxnumber>', methods=['GET'])
def get_search_query(query, maxnumber):
    """
        Search result of query
    """
    app.logger.warning("Falta implementar")
    return ("Json con %s resultados, cada uno con id y audio+desc url,correspondientes con %s" % (maxnumber, query))


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
        list audio files
    """
    app.logger.warning("Falta implementar en formato definitivo")
    outlist = ""
    for subdir, dirs, files in os.walk(DATA_PATH):
        for f in files:
            if os.path.splitext(f)[1] in ext_filter:
                outlist += subdir+'/'+ f + "\n"
    return(outlist)

@auto.doc('public')
@app.route('/list/samples', methods=['GET'])
def list_samples():
    """
        list sample files (segmented pistas)
    """
    app.logger.warning("Falta implementar en formato definitivo")
    outlist = ""
    for subdir, dirs, files in os.walk(SAMPLES_PATH):
        for f in files:
            if os.path.splitext(f)[1] in ext_filter:
                outlist += subdir+'/'+ f + "\n"
    return(outlist)

if __name__ == "__main__":
    file_handler = logging.FileHandler('mock_redpanal_api_ws.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    #app.run()
    app.run( debug=True, host="0.0.0.0", port=5000 )

