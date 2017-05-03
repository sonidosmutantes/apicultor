#ffmpeg -i input.mp3 -af silenceremove=1:0:-50dB output.mp3V,
for i in *.wav; do ffmpeg -i "$i" -af silenceremove=1:0:-50dB nosilence-"${i%.wav}.wav"; done
