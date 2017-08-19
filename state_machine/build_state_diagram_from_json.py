#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import graphviz as gv
import sys

#JSON composition file
diagram_comp = ""
try:
	print("Loading %s"%sys.argv[1])
	json_comp_file = sys.argv[1] 
	# with open(json_file,'r') as file:
	#     json_data = json.load( file )
	diagram_comp = json.load( open(json_comp_file,'r') )
except Exception as e:
	print(e)
	print("JSON composition file error.")
	sys.exit(2)

state_diag = gv.Digraph(format='svg')
num_to_state = dict()
for state in diagram_comp['statesArray']:
    print( "State %s"%state['text'] )
    state_diag.node( state['text'] )
    num_to_state[ state['id'] ] = state['text']

for link in diagram_comp['linkArray']:
    print( "Link from %s to %s with value %s"%(num_to_state[link['from']],num_to_state[link['to']],link['text']) )
    #print( num_to_state[ link['to'] ] )
    state_diag.edge(num_to_state[ link['from'] ], num_to_state[ link['to'] ], link['text'] )

state_diag.render(filename='comp_diag')
