# -*- coding: UTF-8 -*-

"""
    Used by MockRedPanalAPI_service.py
"""

import os
import flask
from flask import jsonify
import logging
import os
import json
from flask import abort
import itertools

# Try to import flask_autodoc, but make it optional
try:
    from flask_autodoc import Autodoc
    HAS_AUTODOC = True
except ImportError:
    HAS_AUTODOC = False
    print("Warning: flask_autodoc not available - API documentation disabled")

#TODO: get from config files
DATA_PATH = "./data"
SAMPLES_PATH = "./samples"

app = flask.Flask(__name__)

# Only initialize Autodoc if available
if HAS_AUTODOC:
    auto = Autodoc(app)
else:
    auto = None

ext_filter = ['.mp3','.ogg','.ogg','.wav']

#
#
#
# TODO: definir una interfaz, luego implementar
#
#
# def get_url_audio(id) (returns an url)
# def get_list_of_files(FILES_PATH) (returns a list separated by \n or a json)
# get_list_of_files_comparing(FILES_PATH, querydescriptor, fixedfloatvalue, comp=">"):
#   (returns a list separated by \n or a json)
# In case of error: abort(404)
#
#
#

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

def get_list_of_sample_files_same_cluster(samplename):
    #TODO: implement. Clusters needs to be defined
    l = list()
    l.append(samplename)
    return l

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
            try:
                #FIXME: lista .json q después no puede abrir (?)
                desc = json.load( open(FILES_PATH + "/" + filename + ".json",'r') )
            # print filename+extension
        # try:
                value = float(desc[querydescriptor])
                if comp==">":
                    if value>comp_value:
                        print(filename+extension, value)
                        # outlist += subdir+'/'+ f + "\n"
                        outlist.append(subdir+'/'+ filename + ".wav") #TODO: check if it's always a wav file (or filter it)
                elif comp=="<":
                    if value<comp_value:
                        print(filename+extension, value)
                        # outlist += subdir+'/'+ f + "\n"
                    outlist.append(subdir+'/'+ filename + ".wav")
            except Exception as e:
                app.logger.error( e )
        # except Exception, e:
        #     app.logger.error( e )
    return outlist