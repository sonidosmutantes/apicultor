#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
import os
import urllib.request

# WARNING: python3

# REST API RedPanal
URL_BASE = "http://127.0.0.1:5000"  # "http://api.redpanal.org.ar"

# OSC Server
DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 5005
DEFAULT_NUMBER_OF_RESULTS = 10

#TODO: add login

def search_handler(unused_addr, args, query):
    call = '/search/%s/%s'%(query,DEFAULT_NUMBER_OF_RESULTS)
    with urllib.request.urlopen(URL_BASE + call) as response:
        print( response.read() )
#search_handler()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default=DEFAULT_IP, help="The ip to listen on")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="The port to listen on")
    args = parser.parse_args()


    dispatcher = dispatcher.Dispatcher()
    #dispatcher.map("/debug", print) #

    dispatcher.map("/search", search_handler, "Search", DEFAULT_NUMBER_OF_RESULTS)

    server = osc_server.ThreadingOSCUDPServer(
              (args.ip, args.port), dispatcher)

    print("Serving on {}".format(server.server_address))
    server.serve_forever()
