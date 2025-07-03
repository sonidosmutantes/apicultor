#!/bin/sh

SERVER_URL="127.0.0.1:5000" 
 
wget http://$SERVER_URL/documentation -O API-Documentation.html
