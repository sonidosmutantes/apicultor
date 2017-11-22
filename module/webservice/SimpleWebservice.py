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

DATA_PATH = "./data"

app = flask.Flask(__name__)
auto = Autodoc(app)

#########################
# Helper functions
#########################


#########################
# API functions
#########################

@app.route('/documentation')
def documentation():
   """
       Doc generation
   """
   return auto.html('public').replace("&amp;lt;br&amp;gt;","") # quick fix with replace (html gen)

soundsdir = "/sounds"
# soundsdir = "/pistas"

@auto.doc('public')
@app.route(soundsdir+'/<int:id>', methods=['GET'])
def get_pista(id):
    """
        Info de la pista (json)
    """
    try:
        desc = json.load( open(DATA_PATH + "/" + str(id) + ".json",'r') )
        return jsonify(desc)
    except:
        abort(404) #not found

@auto.doc('public')
@app.route(soundsdir+'/<int:id>/audio', methods=['GET'])
def get_pista_audio(id):
    """
        Retorna el full path donde esta descargado el audio con ese id
    """
    try:
    	desc = json.load( open(DATA_PATH + "/" + str(id) + ".json",'r') )
    	return flask.send_file( DATA_PATH + "/" + desc['filename'] )
    except:
        abort(404) #not found

if __name__ == "__main__":
    file_handler = logging.FileHandler('mock_api_ws.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    #app.run()
    app.run( debug=True, host="0.0.0.0", port=5000 )
