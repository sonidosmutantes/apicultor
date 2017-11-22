#!/bin/bash

docker start apicultor
docker exec -it apicultor /opt/run_ws.sh
