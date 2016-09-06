s.boot; //start server

//create the buffer
//b = Buffer.alloc(s, s.sampleRate * 8.0, 2); // an 8 second stereo buffer
// vs (comparar) TODO

//TODO: probar las combinaciones de tener en b y c lo mismo. Lugo loopeo uno con el freeze del otro, etc (mismo contenido armónico)
a = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1194_sample1.wav");
b = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/126_sample0.wav");
c = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/982_sample1.wav");
d = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/795_sample1.wav");

//When Buffers are allocated on the server, they're given a unique ID number, so we can reference them later. This number is called the bufnum. It's the second argument to the SynthDef. This way, we know which buffer we're supposed to play

//Buffer.freeAll;

//play synth
SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, player).dup }).add;

//freeze synth
SynthDef(\mutantefreeze, { arg out=0, soundBufnum=0, point=0, vol=1;
    var in, chain;
    in = PlayBuf.ar(1, soundBufnum, BufRateScale.kr(soundBufnum),loop: 1);
    chain = FFT(LocalBuf(4096), in);
    chain = PV_MagFreeze(chain, point);
	Out.ar(out, vol * IFFT(chain).dup);
}).add;


//Trigger sound with pads
MIDIFunc.noteOn({ |veloc, num, chan, src|
	//First row of pads
	        if(num == 48,{
/*		        x.set(\vol, val/127); //volumen 0..1
	  	        (val/127).postln;
		*/
	        	("Pad 48").postln;
	      	//TODO: request a new sound to APICULTOR (MIR, etc)
		        x.free;
		        x = Synth(\mutantefreeze, [\soundBufnum, a]);
			});
			if(num == 49,{
	        	("Pad 49").postln;
		        y.free;
		        y = Synth(\mutantefreeze, [\soundBufnum, b]);
			});
			 if(num == 50,{
	        	("Pad 50").postln;
                z.free;
		        z = Synth(\mutantefreeze, [\soundBufnum, c]);
			});
			if(num == 51,{
	        	("Pad 51").postln;
                q.free;
		        q = Synth(\mutantefreeze, [\soundBufnum, d]);
			});

	//Second row
			if(num == 44,{
	        	("Pad 44").postln;
		        //r = Synth(\playBufMono, [\bufnum, a.bufnum, \rate, 0.5]); //a (half speed)
		        r = Synth(\playBufMono, [\out, 0, \bufnum, a.bufnum, \rate, 1]); //a
		        r = Synth(\playBufMono, [\out, 1, \bufnum, a.bufnum, \rate, 1]); //a
			});
			if(num == 45,{
	        	("Pad 45").postln;
				r = Synth(\playBufMono, [\out, 0, \bufnum, b.bufnum, \rate, 1]); //b
				r = Synth(\playBufMono, [\out, 1, \bufnum, b.bufnum, \rate, 1]); //b
			});
			if(num == 46,{
	        	("Pad 46").postln;
				r = Synth(\playBufMono, [\out, 0, \bufnum, c.bufnum, \rate, 1]); //c
				r = Synth(\playBufMono, [\out, 1, \bufnum, c.bufnum, \rate, 1]); //c
			});
			if(num == 19,{
	        	("Pad 19").postln;
				r = Synth(\playBufMono, [\out, 0, \bufnum, d.bufnum, \rate, 1]); //d
				r = Synth(\playBufMono, [\out, 1, \bufnum, d.bufnum, \rate, 1]); //d
			});
});

MIDIIn.control = {arg src, chan, num, val;
			if(num == 7,{
		        x.set(\vol, val/127); //volumen 0..1
	  	       // (val/127).postln;
			});
			if(num == 10,{
			   x.set(\point, val/127); // PAN value 0..127
		       //(val/127).postln;
			});

			if(num == 8,{
		        y.set(\vol, val/127); //volumen 0..1
	  	        //(val/127).postln;
			});
			if(num == 1,{
			   y.set(\point, val/127); // PAN value 0..127
		       //(val/127).postln;
			});

			if(num == 12,{
		        z.set(\vol, val/127); //volumen 0..1
	  	        //(val/127).postln;
			});
			if(num == 13,{
			   z.set(\point, val/127); // PAN value 0..127
		       //(val/127).postln;
			});

			if(num == 11,{
		        q.set(\vol, val/127); //volumen 0..1
	  	        //(val/127).postln;
			});
			if(num == 33,{
			   q.set(\point, val/127); // PAN value 0..127
		       (val/127).postln;
			});
};

MIDIIn.connectAll;

// Monitor
// //Mostrar MIDI input (controls)
// MIDIIn.control = {arg src, chan, num, val;
// 	[chan,num,val].postln;
// };
//
// //Mostrar nota + velocity
// MIDIFunc.noteOn({ |veloc, num, chan, src|
// 	( "New note received " + num + " with vel "+veloc ).postln;
// });

// Cleanup
// s.quit; //stops server
