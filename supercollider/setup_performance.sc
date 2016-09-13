s.boot; //start server

i= "192.168.56.101"; //APIcultor WebService IP @VM

a = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1194_sample1.wav");
b = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1264_sample0.wav");
c = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/982_sample1.wav");
d = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/795_sample1.wav"); //Variable buffer!

e = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1065_sample.wav" );
//f -> filename then
g =Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/Cuesta_caminar_batero_sample2.wav" );
h=Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/251_sample1.wav" );
i=Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1291_sample2.wav" );

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
    // * Bank1: First ROW of pads *

	//1_A
    if(num == 48,{
/*		        x.set(\vol, val/127); //volumen 0..1
	  	        (val/127).postln;
		*/
	        	("Pad 48").postln;
	      	//TODO: request a new sound to APICULTOR (MIR, etc)
		        x.free;
		        x = Synth(\mutantefreeze, [\soundBufnum, a]);
	});

	//B
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

	// * Bank1: Second ROW of pads *
	//E?
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

				m = 15; //Length of sound list (TODO: retrieve from API) FIXME

		        //Get a new sample file from apicultor (HFC < 1 )
		        format("curl http://%:5000/search/mir/samples/HFC/lessthan/1000/% -o desc.tmp", i,m).unixCmd; //mac os
		       //FIXME: wait to download here? (takes effect next time)

		        f = FileReader.read("./desc.tmp".standardizePath); //array

		        v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
		        v.postln(); //selected file
		        f = ("/Users/hordia/Documents/vmshared"+v.replace("./","/")).replace(" ",""); //trim spaces (TODO: check why there is an extra space in the path)
		        c = Buffer.read(s, f );


		        //plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, c.bufnum, \rate, 1]); //c
				r = Synth(\playBufMono, [\out, 1, \bufnum, c.bufnum, \rate, 1]); //c
			});

	        //Surprise sound (TODO: add MIR parameter)
			if(num == 19,{

	        	("Pad 19").postln;

				m = 15; //Length of sound list (TODO: retrieve from API) FIXME

		        //Get a new sample file from apicultor HFC>40
		        format("curl http://%:5000/search/mir/samples/HFC/greaterthan/40000/% -o desc.tmp", i,m).unixCmd; //mac os
		       //FIXME: wait to download here? (takes effect next time)

		        f = FileReader.read("./desc.tmp".standardizePath); //array

		        v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
		        v.postln(); //selected file
		        f = ("/Users/hordia/Documents/vmshared"+v.replace("./","/")).replace(" ",""); //trim spaces (TODO: check why there is an extra space in the path)
		        d = Buffer.read(s, f );


		        //plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, d.bufnum, \rate, 1]); //d
				r = Synth(\playBufMono, [\out, 1, \bufnum, d.bufnum, \rate, 1]); //d
			});

    // * Bank2: First ROW of pads *

	//2_A
    if(num == 36,{
	    ("Pad 46").postln;
				//plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, e.bufnum, \rate, 1]); //e @ L channel
				r = Synth(\playBufMono, [\out, 1, \bufnum, e.bufnum, \rate, 1]); //e @ R channel
	});

	//B
	if(num == 37,{
	        	("Pad 37").postln;
				//plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, g.bufnum, \rate, 1]); //g @ L channel
				r = Synth(\playBufMono, [\out, 1, \bufnum, g.bufnum, \rate, 1]); //g @ R channel
	});
	if(num == 38,{
	        	("Pad 38").postln;
						//plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, h.bufnum, \rate, 1]); //h @ L channel
				r = Synth(\playBufMono, [\out, 1, \bufnum, h.bufnum, \rate, 1]); //h @ R channel
	});
	if(num == 39,{
	        	("Pad 39").postln;
						//plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, i.bufnum, \rate, 1]); //i @ L channel
				r = Synth(\playBufMono, [\out, 1, \bufnum, i.bufnum, \rate, 1]); //i @ R channel
	 });


	// * Bank2: Second ROW of pads *

	//
    if(num == 32,{
	    ("Pad 32").postln;
	});

	//
	if(num == 33,{
	        	("Pad 33").postln;
	});
	if(num == 34,{
		("Pad 34").postln;
	});
	if(num == 35,{
	        	("Pad 35").postln;
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

// Cleanup
// s.quit; //stops server
