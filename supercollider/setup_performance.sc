s.boot; //start server

//----------------------
//----------------------

i= "192.168.56.101"; //APIcultor WebService IP @VM

//TODO configure L & R channels (different synths or configuration (via midi)
//TODO: add multichannel support

~bank1a = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1194_sample1.wav");
~bank1b = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1264_sample0.wav");
~bank1c = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/982_sample1.wav");
~bank1d = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/795_sample1.wav"); //Variable buffer!


~bank2a= Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1065_sample.wav" );
//f -> filename then
g =Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/Cuesta_caminar_batero_sample2.wav" );
h=Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/251_sample1.wav" );
i=Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1291_sample2.wav" );


//--- synths
//play synth
SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, player).dup }).add;

/*
Testing PAN

SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1, pan = 2, channels=2 |  var scaledRate, player;
	scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2); Out.ar(out, Mix.ar(PanAz.ar(2, [player,player], [1,-1+.3]))).dup  }).add; //pan range 0..1 (mapea a -1..1)*/

// SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1, pan = 1, channels=2 |  var scaledRate, player;
// scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2); Out.ar(out, Mix.ar(PanAz.ar(2, [player, player], [pan, pan+0.3]))).dup  }).add;
//
// SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1, pan = 1, channels=2 |  var scaledRate, player;
// scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2); Out.ar(out, Mix.ar(PanAz.ar(2, player, [pan, pan+0.3]))).dup  }).add;
//
// SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1, pan = 0, channels=2 |  var scaledRate, player;
// scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2); Out.ar(out, Mix.ar(PanAz.ar(2, player, [-1+pan, pan]))).dup  }).add;
//
// PanAz.ar(
// 	5, 				// numChans
// 	ClipNoise.ar, 	// in
// 	LFSaw.kr(MouseX.kr(0.2, 8, 'exponential')), // pos
// 	0.5,			// level
// 	3			// width
// );


//freeze synth
SynthDef(\mutantefreeze, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0;
    var in, chain;
    in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
    chain = FFT(LocalBuf(4096), in);
    chain = PV_MagFreeze(chain, point);
	//Out.ar(out, vol * IFFT(chain).dup);
	Out.ar(out, Pan2.ar(vol * IFFT(chain).dup, pan));
}).add;

~fftwidth = 4096;
/*~fftwidth = 512;
~fftwidth = 1024;
~fftwidth = 2048;
~fftwidth = 4096;
~fftwidth = 8192;*/

~channel = 0;
// ~channel = 1;
~rate = 1; //normal
// ~rate = 0.5; //half speed

//Trigger sound with pads
MIDIFunc.noteOn({ |veloc, num, chan, src|
    // * Bank1: First ROW of pads *
    if(num == 48,{
	        	("Bank1 / A / Pad 48").postln;
		        x.free;
		        x = Synth(\mutantefreeze, [\bufnum, ~bank1a, \out, ~channel, \fftwidth, ~fftwidth]);
	});
	if(num == 49,{
	        	("Bank1 / B / Pad 49").postln;
		        y.free;
		        y = Synth(\mutantefreeze, [\bufnum, ~bank1b, \out, ~channel]);
	});
	if(num == 50,{
	        	("Bank1 / C / Pad 50").postln;
                z.free;
		        z = Synth(\mutantefreeze, [\bufnum, ~bank1c, \out, ~channel]);
	});
	if(num == 51,{
	        	format("Bank1 / D / Pad %",num).postln;
                q.free;
		        q = Synth(\mutantefreeze, [\bufnum, ~bank1d, \out, ~channel]);
	 });

	// * Bank1: Second ROW of pads *
	if(num == 44,{
	        //	("Pad 44").postln;
		    //Synth(\playBufMono, [\out, 0, \bufnum, ~bank1a, \rate, ~rate, \out, ~channel]);
            //x.get(\point, { arg value; ("point is now:" + value).postln; });
            x.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		x.set(\point, 0);
					("Bank1 / A / freeze OFF").postln;
	    		}, {
					 x.set(\point, 1);
					("Bank1 / A / freeze ON").postln;
			    });
			});
	});
			if(num == 45,{
	        	("Pad 45").postln;
				Synth(\playBufMono, [\out, 0, \bufnum, ~bank1b, \rate, ~rate, \out, ~channel]);
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
		        //x.set(\point, val/127); // point 0..1
		        //(val/127*8192).postln;
	         	//if(val/127>0.5,x.set(\out, 1), x.set(\out, 0) ); //TODO: ver
		        if(val/127>0.5,x.set(\pan, -1), x.set(\pan, 0) ); //TODO: ver
			});

			if(num == 8,{
		        y.set(\vol, val/127); //volumen 0..1
			});
			if(num == 1,{
			   y.set(\point, val/127); // point 0..1
			});

			if(num == 12,{
		        z.set(\vol, val/127); //volumen 0..1
			});
			if(num == 13,{
			   z.set(\point, val/127); // point 0..1
			});

			if(num == 11,{
		        q.set(\vol, val/127); //volumen 0..1

			});
			if(num == 33,{
			   q.set(\point, val/127); // point 0..1
			});
};

MIDIIn.connectAll;

// Cleanup
// s.quit; //stops server
