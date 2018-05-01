
#raspberry
alias ffmpeg=avconv
export DISPLAY=:0.0

echo "Disable debug mode in .py or config"
python CloudInstrument.py >>apicultor.log 2>&1 &
#python CloudInstrument.py >/dev/null 2>&1 &
#python CloudInstrument.py >/dev/null &

#scsynth -u  57120 &
sclang -D apicultor_synth.scd
