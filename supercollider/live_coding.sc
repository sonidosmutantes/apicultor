i.postln; //apicultor ws ip

Buffer.freeAll;

MIDIIn.connectAll;

r = Synth(\playBufMono, [\bufnum, a.bufnum, \rate, 0.5]); //buffer a at half speed