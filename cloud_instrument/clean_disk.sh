#cd ~/dev/apicultor/cloud_instrument

STARTUP_FILES="FILE1.wav FILE2.wav"

mkdir startupfiles
mv $STARTUP_FILES startupfiles/
rm *.wav* *.aif* *.ogg *.mp3 *.mp4
mv startupfiles/* .

#rm apicultor.log
