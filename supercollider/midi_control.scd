//mapeo 0..127 a 200..300
//w=val.linlin(0,127,200,300)})

MIDIIn.connectAll;

//Trigger sound with pads
MIDIFunc.noteOn({ |veloc, num, chan, src|

	// * Bank1: First ROW of pads *
    if(num == 48,{
/*	        	("Bank1 / A / Pad 48").postln;
		        x.free;
		        x = Synth(\mutantefreeze, [\bufnum, ~bank1a, \out, ~channel, \fftwidth, ~fftwidth]);*/

			        ("Pad 44").postln;
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
	if(num == 49,{
	        y.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		y.set(\point, 0);
					("Bank1 / B / freeze OFF").postln;
	    		}, {
					 y.set(\point, 1);
					("Bank1 / B / freeze ON").postln;
			    });
			});
	});
	if(num == 50,{
	        z.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		z.set(\point, 0);
					("Bank1 / C / freeze OFF").postln;
	    		}, {
					 z.set(\point, 1);
					("Bank1 / C / freeze ON").postln;
			    });
			});
	});
	if(num == 51,{
	        q.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		q.set(\point, 0);
					("Bank1 / D / freeze OFF").postln;
	    		}, {
					 q.set(\point, 1);
					("Bank1 / D / freeze ON").postln;
			    });
			});
	 });

    // * Bank1: Second ROW of pads *
	if(num == 44,{
	        ("Pad 44").postln;
	        g.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		g.set(\point, 0);
					("Bank1 / E / freeze OFF").postln;
	    		}, {
					 g.set(\point, 1);
					("Bank1 / E / freeze ON").postln;
			    });
			});
	});
	if(num == 45,{
	        ("Pad 45").postln;
	        h.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		h.set(\point, 0);
					("Bank1 / F / freeze OFF").postln;
	    		}, {
					 h.set(\point, 1);
					("Bank1 / F / freeze ON").postln;
			    });
			});
	});
	if(num == 46,{
	        ("Pad 46").postln;
	        i.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		i.set(\point, 0);
					("Bank1 / G / freeze OFF").postln;
	    		}, {
					 i.set(\point, 1);
					("Bank1 / G / freeze ON").postln;
			    });
			});
    });
	if(num == 19,{
	     ("Pad 46").postln;
	     j.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
			 		j.set(\point, 0);
					("Bank1 / H / freeze OFF").postln;
	    		}, {
					 j.set(\point, 1);
					("Bank1 / H / freeze ON").postln;
			    });
			});
	});

    // * Bank2: First ROW of pads *

	//2_A
    if(num == 36,{
	    ("Pad 36").postln;
	//Surprise sound (TODO: add MIR parameter)
		m = 15; //Length of sound list (TODO: retrieve from API) FIXME

		//Get a new sample file from apicultor (HFC < 1 )
		format("curl http://%:5000/search/mir/samples/HFC/lessthan/1000/% -o /Users/hordia/desc.tmp", ~ip,m).unixCmd; //mac os

		//FIXME: wait to download here? (takes effect next time)

		f = FileReader.read("/Users/hordia/desc.tmp".standardizePath); //array
        v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
        v.postln(); //selected file
        f = ("/Users/hordia/Documents/vmshared"+v.replace("./","/")).replace(" ",""); //trim spaces (TODO: check why there is an extra space in the path)

        a = Buffer.read(s, f ); // new buffer A

		//plays new sample (two channels)
		//w.free;
		//w = ~prepare_freeze.value(~bank1a, ~channel, "Bank1", "A");
        x = Synth(\mutantefreeze, [\bufnum, a, \out, ~channel, \vol, 0]);
		        //plays new sample
				//plays new sample
		        ~speaker1 = 0;
				~speaker2 = 1;
		        //r = Synth(\playBufMono, [\out, ~speaker0, \bufnum, a.bufnum, \rate, 1]); //e @ L channel
				r = Synth(\playBufMono, [\out, ~speaker1, \bufnum, a, \rate, 1]); //e @ R channel
	x.free;
		r.free;
	});
	//B
	if(num == 37,{
	        	("Pad 37").postln;

				m = 15; //Length of sound list (TODO: retrieve from API) FIXME

		//Get a new sample file from apicultor (duration < 1 )
		format("curl http://%:5000/search/mir/samples/duration/lessthan/1000/% -o /Users/hordia/desc.tmp", ~ip,m).unixCmd; //mac os

		//FIXME: wait to download here? (takes effect next time)

		f = FileReader.read("/Users/hordia/desc.tmp".standardizePath); //array
        v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
        v.postln(); //selected file
        f = ("/Users/hordia/Documents/vmshared"+v.replace("./","/")).replace(" ",""); //trim spaces (TODO: check why there is an extra space in the path)

        a = Buffer.read(s, f ); // new buffer A

		//plays new sample (two channels)
		y = Synth(\mutantefreeze, [\bufnum, a, \out, ~channel, \vol, 0]);
		        //plays new sample
				//plays new sample
		        ~speaker1 = 0;
				~speaker2 = 1;
		        r = Synth(\playBufMono, [\out, ~speaker1, \bufnum, a, \rate, 21.5]); //e @ L channel
				r = Synth(\playBufMono, [\out, ~speaker2, \bufnum, a, \rate, 0.1]); //e @ R channel
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

/*MIDIIn.control = {arg src, chan, num, val;
			if(num == 7,{
		        x.set(\vol, val/127); //volumen 0..1
	  	       // (val/127).postln;
			});
			if(num == 10,{
		        //x.set(\point, val/127); // point 0..1
		        //(val/127*8192).postln;
	         	//if(val/127>0.5,x.set(\out, 1), x.set(\out, 0) ); //TODO: ver

		        if(val/127>0.5,x.set(\pan, -1), x.set(\pan, 0) ); //TODO: ver

		        //~newStart = (~bank1a.numFrames / ~bank1a.numChannels )*val/127;
		        //x.set(\startPos, ~newStart); //freeze start position


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
};*/

~maxvol = 3;
MIDIIn.control = {arg src, chan, num, val;
	//[chan,num,val].postln; //monitor

//First row of knobs
<<<<<<< HEAD:supercollider/midi_control.scd
            if(num == 7,{ //VOL W
		        x.set(\vol, val.linlin(0,127,0,~maxvol+4)); //volumen 0..1

			});
			if(num == 10,{ //VOL X
			    y.set(\vol, val.linlin(0,127,0,~maxvol)); // point 0..1
			});
			if(num == 8,{//VOL Y
		        z.set(\vol, val.linlin(0,127,0,~maxvol)); //volumen 0..1
			});
			if(num == 1,{//VOL Z
			   q.set(\vol, val.linlin(0,127,0,1)); // point 0..1 -> Fix 1.0
=======
/*			if(num == 7,{ //PAN W
		        //w.set(\vol, val/127); //volumen 0..1
	  	       // (val/127).postln;
		 ("PAN W").postln;
			});
			if(num == 10,{ //PAN X
		        //x.set(\point, val/127); // point 0..1
		        //(val/127*8192).postln;
	         	//if(val/127>0.5,x.set(\out, 1), x.set(\out, 0) ); //TODO: ver

	         	("PAN X").postln;
		        if(val/127>0.5,x.set(\pan, -1), x.set(\pan, 0) ); //TODO: ver //FIXME

		        //~newStart = (~bank1a.numFrames / ~bank1a.numChannels )*val/127;
		        //x.set(\startPos, ~newStart); //freeze start position
			});
			if(num == 8,{
		        //y.set(\vol, val/127); //volumen 0..1
		        ("PAN Y").postln;
			});
			if(num == 1,{
			   //y.set(\point, val/127); // point 0..1
		 ("PAN Z").postln;
			});*/
            if(num == 7,{ //VOL x
		        x.set(\vol, val/127); //volumen 0..1
			});
			if(num == 10,{ //VOL y
			    y.set(\vol, val/127); // point 0..1
			});
			if(num == 8,{//VOL z
		//z.set(\volmax, 5);
		         //FIXME: ~volmax = z.get(\volmax, {arg a; a.postln() });
		         ~volmax = 5;
		         z.set(\vol, val/127*~volmax); //volumen 0..1
			});
			if(num == 1,{//VOL q
			   q.set(\vol, val/127); // point 0..1
>>>>>>> c81a747a9a86c4b8ce6b05ddb5bac63ad99301d1:supercollider/setup_performance.sc
			});


//Second row of knobs
			if(num == 12,{ //VOL W
		        g.set(\vol, val.linlin(0,127,0,~maxvol)); //volumen 0..1
			});
			if(num == 13,{ //VOL X
			   h.set(\vol, val.linlin(0,127,0,~maxvol)); // point 0..1
			});
			if(num == 11,{//VOL Y
		       i.set(\vol, val.linlin(0,127,0,~maxvol)); //volumen 0..1
			});
			if(num == 33,{//VOL Z
			   j.set(\vol, val.linlin(0,127,0,~maxvol)); // point 0..1
			});
};

MIDIIn.connectAll;

// Monitor

/*
//Mostrar MIDI input (controls)
MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln;
};

//Mostrar nota + velocity
MIDIFunc.noteOn({ |veloc, num, chan, src|
	( "New note received " + num + " with vel "+veloc ).postln;
});
*/