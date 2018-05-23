export DISPLAY=:0.0

# Cloud instrument path
cd ~/dev/apicultor/cloud_instrument

#APICultor service
lxterminal -t CloudService -e python CloudInstrument.py >/dev/null & #no log
#python CloudInstrument.py >apicultor.log  2>&1 &

# MIDI autoconnect (Yaeltex MIDI ctrl return)
/home/pi/autoconnect-midi.sh >/dev/null 2>&1 &
sleep 1
/home/pi/autoconnect-midi.sh >/dev/null 2>&1 &


#Supercollider apicultor synth (LCD 3.5')
lxterminal -t API.Cultor --geometry=65x20 -e sclang -D apicultor_synth.scd # lcd 3.5'

#Supercollider apicultor synth (LCD 5')
#lxterminal -t API.Cultor --geometry=75x35 -e sclang -D apicultor_synth.scd # 5'
