#!/bin/bash
PATH=/usr/local/bin:$PATH
export DISPLAY=:0.0
sleep 5  #can be lower (5) for rpi3

# WARNING. Consume recursos. Mejor correrlo en otra pac
# Open Stage Control
#cd bin/open-stage-control-0.28.2-linux-armv7l 
#./open-stage-control -l /home/pi/dev/apicultor/cloud_instrument/ui/arcitec.json -s 127.0.0.1:57120 -p 8080 -o 7000 --no-gui &


#Common version
#~/apicultor_cloud_instrument.sh

#GUI version
~/apicultor_gui.sh

