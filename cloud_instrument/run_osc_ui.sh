UI=nube_sinte_simple
#UI=nube_sinte_complex
# with -n -> no external UI (browser only)
#open-stage-control -n -l ui/$UI.js -s 127.0.0.1:5555 127.0.0.1:9001
open-stage-control -l ui/$UI.js -s 127.0.0.1:9001 -p 7000

#$ open-stage-control -s 127.0.0.1:5555 127.0.0.1:6666 -p 7777
#
#This will create an app listening on port 7777 for synchronization messages, and sending its widgets state changes to ports 5555 and 6666.
