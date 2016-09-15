s.boot; //start server

MIDIIn.connectAll;

// Monitor: Mostrar nota + velocity
//MIDIFunc.noteOn({ |veloc, num, chan, src|
//	( "New note received " + num + " with vel "+veloc ).postln;
//});

// In windows scape with \\
a = Buffer.read(s, "C:\\Users\\hordia\\git\\apicultor\\samples\\126_sample0.wav");

//play synth
SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, player).dup }).add;


//Trigger sound with pads
MIDIFunc.noteOn({ |veloc, num, chan, src|
    if(num == 30,{
        	    ("Pad 30").postln;
		        Synth(\playBufMono, [\bufnum, a]);
		        //Synth(\playBufMono, [\bufnum, a, \out, 1]);
	});
});

// Cleanup
// s.quit; //stops server
