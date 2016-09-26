s.boot; //start server


MIDIIn.connectAll;

// Monitor
//Mostrar nota + velocity
MIDIFunc.noteOn({ |veloc, num, chan, src|
	( "New note received " + num + " with vel "+veloc ).postln;
});
//Mostrar MIDI input (controls)
MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln;
};


// In windows scape with \\
b = Buffer.read(s, "C:\\Users\\hordia\\git\\apicultor\\samples\\drums\\snare-punch.wav");
a = Buffer.read(s, "C:\\Users\\hordia\\git\\apicultor\\samples\\drums\\crash-tape.wav");
c = Buffer.read(s, "C:\\Users\\hordia\\git\\apicultor\\samples\\1194_sample3.wav");
//a.free;

//play synth
SynthDef(\playBufMono, {| out = 0, bufnum = 0, vol=1, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, vol * player).dup }).add;


//Trigger sound with pads
MIDIFunc.noteOn({ |veloc, num, chan, src|
    if(num == 30,{
        	    ("Pad 30").postln;
		        Synth(\playBufMono, [\bufnum, a, \vol, veloc/127]);
		        //Synth(\playBufMono, [\bufnum, a, \out, 1]);
	});
    if(num == 46,{
        	    ("Pad 46").postln;
		        Synth(\playBufMono, [\bufnum, a, \vol, veloc/127]);
		        //Synth(\playBufMono, [\bufnum, a, \out, 1]);
	});

});

MIDIIn.control = {arg src, chan, num, val;
			if(num == 7,{
		        r.set(\vol, val/127); //volumen 0..1
	  	       (val/127).postln;
			});
}
// Cleanup
// s.quit; //stops server

//playing
r = Synth(\playBufMono, [\bufnum, c.bufnum, \rate, 1]); //buffer c
r = Synth(\mutantefreeze, [\bufnum, c.bufnum, \rate, 1]); //
r.free;
