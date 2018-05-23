# aconnect -l (lista los puertos)


while ! aconnect SuperCollider:4 'MIDI Mix':0; do sleep 1; done # Akai MIDIMix Controller
#while ! aconnect SuperCollider:4 'API-cultor':0; do sleep 1; done # Yaeltex MIDI ctrl return
