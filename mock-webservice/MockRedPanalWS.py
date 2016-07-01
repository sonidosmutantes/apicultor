#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import flask
from flask import jsonify
from flask_autodoc import Autodoc
import logging

DATA_PATH = "./data/"

app = flask.Flask(__name__)
auto = Autodoc(app)


@app.route('/documentation')
def documentation():
    """
        Doc generation
    """
    return auto.html('public')


@auto.doc('public')
@app.route('/pistas/<int:id>/audio', methods=['GET'])
def get_pista_audio(id):
    app.logger.warning("Falta implementar")
    return("URL del audio ID %s" % id)


@auto.doc('public')
@app.route('/pistas/<int:id>/descriptor', methods=['GET'])
def get_descriptor_audio(id):
    app.logger.warning("Falta implementar")
    return("Descriptor json del audio ID %s" % id)


@auto.doc('public')
@app.route('/pistas/<int:id>', methods=['GET'])
def get_pista(id):
    """
        json con path a archivos de audio y del descriptor
    """
    app.logger.warning("Falta implementar")
    return jsonify(id=id,
                   desc=DATA_PATH + str(id) + ".json",
                   audio=DATA_PATH + str(id) + ".ogg")


@app.route('/search/<query>/<int:number>', methods=['GET'])
def get_search_query(query, number):
    app.logger.warning("Falta implementar")
    return ("Json con %s resultados, cada uno con id y audio+desc url,correspondientes con %s" % (number, query))


@app.route('/search/last/<int:number>', methods=['GET'])
def get_last_searchs(number):
    app.logger.warning("Falta implementar")
    return ("Json con las últimas %i búsquedas" % (int(number)))

@app.route('/search/tag/<tag1>', methods=['GET'])
def get_tag_search(tag1):
    app.logger.warning("Falta implementar")
    return ("Resultado de buscar por tag %s" % (tag1))

@app.route('/list/pistas', methods=['GET'])
def list_pistas():
    app.logger.warning("Falta implementar")
    #list audio files
    ext_filter = ['.mp3','.ogg','.ogg']
    for subdir, dirs, files in os.walk(DATA_PATH):
        for f in files:
            if os.path.splitext(f)[1] in ext_filter:
                print(f)


if __name__ == "__main__":
    file_handler = logging.FileHandler('app.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    
    #app.run()
    app.run(debug=True)
