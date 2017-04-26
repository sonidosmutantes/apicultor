#FIXME for i in *.wav; do mv "$i" "${nosilence-normalized-i%}"; done
for i in *.json; do mv "$i" nosilence-normalized-"${i%.json}".json; done
