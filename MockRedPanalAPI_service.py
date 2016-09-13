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

ext_filter = ['.mp3','.ogg','.ogg','.wav']

#########################
# Helper functions
#########################

def get_url_audio(id):
    for subdir, dirs, files in os.walk(DATA_PATH):
        for f in files:
            if os.path.splitext(f)[1] in ext_filter and os.path.splitext(f)[0]==str(id):
                    return( os.path.abspath(DATA_PATH) + "/" + str(f) )
    abort(404) #not found

def get_list_of_files(FILES_PATH):
    outlist = ""
    for subdir, dirs, files in os.walk(FILES_PATH):
        for f in files:
            if os.path.splitext(f)[1] in ext_filter:
                outlist += subdir+'/'+ f + "\n"
    return(outlist)

#TODO: refactorizar para pasarle a la función un "comparator" en un objeto (para no duplicar código)
def get_list_of_files_comparing(FILES_PATH, querydescriptor, fixedfloatvalue, comp=">"):

    comp_value = float(fixedfloatvalue)/1000. # convierte al valor real desde el valor en punto fijo

    #TODO: agregar el resto de los descriptores soportados
    if querydescriptor=="HFC":
        querydescriptor = "lowlevel.hfc.mean"
    elif querydescriptor=="duration":
        # FIXME: por ej el duration no tiene sentido calcularle el 'mean'
        querydescriptor = "metadata.duration.mean"
    else:
        app.logger.error( "Todavía no implementado" )
        abort(405) #405 - Method Not Allowed

    outlist = list()
    for subdir, dirs, files in os.walk(FILES_PATH):
        for f in files:
            filename, extension = os.path.splitext(f)
            if extension!=".json":
                continue
            desc = json.load( open(FILES_PATH + "/" + filename + ".json",'r') )

            # print filename+extension
        # try:
            value = float(desc[querydescriptor])
            if comp==">":
                if value>comp_value:
                    print filename+extension, value
                    # outlist += subdir+'/'+ f + "\n"
                    outlist.append(subdir+'/'+ filename + ".wav") #TODO: check if it's always a wav file (or filter it)
            elif comp=="<":
                if value<comp_value:
                    print filename+extension, value
                    # outlist += subdir+'/'+ f + "\n"
                    outlist.append(subdir+'/'+ filename + ".wav")
        # except Exception, e:
        #     app.logger.error( e )
    return outlist


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
@app.route('/search/mir/samples/<querydescriptor>/greaterthan/<int:fixedfloatvalue>/<int:maxnumber>', methods=['GET'])
def get_search_mir_query_greater(querydescriptor, fixedfloatvalue, maxnumber):
    """
        Search result of query (mayor)
        Falta implmementar el formato json, por ahora es una lista!
        JSON con %i pistas con el parxmetro %s mayor que %f" % (maxnumber,querydescriptor,comp_value)
    """
    outlist = get_list_of_files_comparing(SAMPLES_PATH, querydescriptor, fixedfloatvalue, ">")
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
    outlist = get_list_of_files_comparing(SAMPLES_PATH, querydescriptor, fixedfloatvalue, "<")
    top5 = itertools.islice(outlist, maxnumber)
    # TODO: Falta implmementar el formato json, por ahora es una lista!
    #       o dar como opción PLAIN/JSON, en plano es más cómodo para laburar en SuperCollider?
    output = ""
    for f in top5:
        output += f + "\n"
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
    return( get_list_of_files(DATA_PATH) )

@auto.doc('public')
@app.route('/list/samples', methods=['GET'])
def list_samples():
    """
        list sample files (segmented pistas)
    """
    app.logger.warning("Falta implementar en formato definitivo") #todos? o poner un máximo?
    return( get_list_of_files(SAMPLES_PATH) )

if __name__ == "__main__":
    file_handler = logging.FileHandler('mock_redpanal_api_ws.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    #app.run()
    app.run( debug=True, host="0.0.0.0", port=5000 )

