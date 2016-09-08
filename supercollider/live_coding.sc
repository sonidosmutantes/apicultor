s.boot; //start server
//s.quit; //stops server

i.postln; //apicultor ws ip

Buffer.freeAll; // no sound

MIDIIn.connectAll;

r = Synth(\playBufMono, [\bufnum, a.bufnum, \rate, 0.5]); //buffer a at half speed


//-----

//Get a new sample in a new buffer
		        //Get a new sample file from apicultor
		        format("curl http://%:5000/list/samples -o desc.tmp", i).unixCmd; //mac os
		       //FIXME: wait to download here? (takes effect next time)

		        f = FileReader.read("./desc.tmp".standardizePath); //array
		        m = 10; //Length of sound list (TODO: retrieve from API) FIXME
		        v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
		        v.postln(); //selected file
		        f = ("/Users/hordia/Documents/vmshared"+v.replace("./","/")).replace(" ",""); //trim spaces (TODO: check why there is an extra space in the path)
		        d = Buffer.read(s, f );


		        //plays new sample
				r = Synth(\playBufMono, [\out, 0, \bufnum, d.bufnum, \rate, 1]); //d
				r = Synth(\playBufMono, [\out, 1, \bufnum, d.bufnum, \rate, 1]); //d

//------

// Monitor

//Mostrar MIDI input (controls)
MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln;
};

//Mostrar nota + velocity
MIDIFunc.noteOn({ |veloc, num, chan, src|
	( "New note received " + num + " with vel "+veloc ).postln;
});
