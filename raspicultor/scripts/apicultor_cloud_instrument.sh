export DISPLAY=:0.0

# Cloud instrument path
cd ~/dev/apicultor/cloud_instrument

#APICultor service
python CloudInstrument.py >/dev/null & #no log
#python CloudInstrument.py >apicultor.log  2>&1 &

# MIDI autoconnect (Yaeltex MIDI ctrl return)
/home/pi/autoconnect-midi.sh >/dev/null 2>&1 &
sleep 1
/home/pi/autoconnect-midi.sh >/dev/null 2>&1 &

#Supercollider apicultor synth
sclang -D apicultor_synth.scd 
