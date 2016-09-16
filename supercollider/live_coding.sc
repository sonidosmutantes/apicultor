s.boot; //start server
//s.quit; //stops server

i.postln; //apicultor ws ip

Buffer.freeAll; // no sound

MIDIIn.connectAll;

r = Synth(\playBufMono, [\bufnum, a.bufnum, \rate, 0.5]); //buffer a at half speed

//g =Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/Cuesta_caminar_batero_sample2.wav" );
i=Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1291_sample2.wav" );
r = Synth(\playBufMono, [\bufnum, i.bufnum, \rate, 0.5]); //buffer a at half speed
-------

//Panning
		LinPan2.ar(in, pos, level);
		LinPan2.ar(in, -1, 1);

play({ LinPan2.ar(PinkNoise.ar(0.4), FSinOsc.kr(2)) }); //oscilating pink noise

SynthDef("help-LinPan2", {  Out.ar(0, LinPan2.ar(FSinOsc.ar(800, 0, 0.1), FSinOsc.kr(3))) }).play;

// You'll only hear the front two channels on a stereo setup.
(
SynthDef("help-Pan4", {
    Out.ar(0, Pan4.ar(PinkNoise.ar, FSinOsc.kr(2), FSinOsc.kr(1.2), 0.3))
}).play;
)

play({ Pan4.ar(PinkNoise.ar, -1,  0, 0.3) }); // left pair
play({ Pan4.ar(PinkNoise.ar,  1,  0, 0.3) }); // right pair
play({ Pan4.ar(PinkNoise.ar,  0, -1, 0.3) }); // back pair
play({ Pan4.ar(PinkNoise.ar,  0,  1, 0.3) }); // front pair

play({ Pan4.ar(PinkNoise.ar,  0,  0, 0.3) }); // center

----
//loop and freeze del sonido 3(c)
volumenes y parametros
				z.set(\vol, 0);
        z.set(\vol, 1);
z.set(\vol, 2);
z.set(\vol, 3);
//z.set(
//-----
// **** Get new sound, fill de buffer and  play ***

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
