s.boot; //start server

a = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1194_sample1.wav");
b = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1264_sample0.wav");
c = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/982_sample1.wav");
d = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/795_sample1.wav"); //Variable buffer!

//play synth
SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, player).dup }).add;


//Trigger sound with pads
MIDIFunc.noteOn({ |veloc, num, chan, src|
    if(num == 48,{
        	    ("Pad 48").postln;
		        x.free;
		        x = Synth(\playBufMono, [\soundBufnum, a]);
	});
	if(num == 49,{
	        	("Pad 49").postln;
		        y.free;
		        y = Synth(\playBufMono, [\soundBufnum, b]);
	});
	if(num == 50,{
	        	("Pad 50").postln;
                z.free;
		        z = Synth(\playBufMono, [\soundBufnum, c]);
	});
	if(num == 51,{
	        	("Pad 51").postln;
                q.free;
		        q = Synth(\playBufMono, [\soundBufnum, d]);
	 });
});

MIDIIn.connectAll;

// Cleanup
// s.quit; //stops server
