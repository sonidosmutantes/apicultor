#!/usr/bin/python2
# -*- coding: UTF-8 -*-

import flask

app = flask.Flask(__name__)

#TODO: add login


@app.route('/pistas/<int:id>/audio', methods=['GET'])
def get_pista_audio(id):
    return("URL del audio ID %s" % id)


@app.route('/pistas/<int:id>/descriptor', methods=['GET'])
def get_descriptor_audio(id):
    return("Descriptor json del audio ID %s" % id)


@app.route('/search/<query>/<int:number>', methods=['GET'])
def get_search_query(query, number):
    return ("Json con %s resultados, cada uno con id y audio+desc url,correspondientes con %s" % (number, query))


@app.route('/search/last/<int:number>', methods=['GET'])
def get_last_searchs(number):
    return ("Json con las últimas %i búsquedas" % (int(number)))


@app.route('/search/tag/<tag1>', methods=['GET'])
def get_last_searchs(tag1):
    return ("Resultado de buscar por tag %s" % (tag1)


if __name__ == "__main__":
    app.run()
