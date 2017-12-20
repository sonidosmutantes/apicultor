/*
    Sounds from the web (using API-Cultor) by hordia
*/

//s.options.sampleRate= 48000; //traktor 2
s.boot; //start server

/*
// .wav record
thisProcess.platform.recordingsDir//locate folder
s.recHeaderFormat;//request default format
s.recHeaderFormat = "wav";
//s.recHeaderFormat = "aiff";
s.record;
s.stopRecording;
*/

/*
*/
// control sending OSC to supercollider service
//OSC address and port
//NetAddr.localAddr //check '127.0.0.1', 57120
~netaddr = NetAddr("127.0.0.1", 57120)
~voice_number = 0; // [0..7]
~hardcoded_file = "/Users/hordia/samples/freesound-sessions/BREATHE_IN_2.wav";
~netaddr.sendMsg("/playfreeze", ~hardcoded_file,~voice_number); // (method, file_path, voice_number)

// control sending OSC to apicultor server
~apicultorsrv = NetAddr("127.0.0.1", 9001);

~voice = 2;
~apicultorsrv.sendMsg("/set_voices", ~voice, 1); //voice 2, en on (1)
~apicultorsrv.sendMsg("/mir/spectral_centroid/mean", 900);
~apicultorsrv.sendMsg("/mir/spectral_complexity/mean", 0.12);
~apicultorsrv.sendMsg("/mir/duration", 3.4); //tip: misma duración, mantengo tempo
~apicultorsrv.sendMsg("/mir/pitch_centroid/mean", 0.78);
~apicultorsrv.sendMsg("/retrieve", 1); //get a new sound

//controls and effects
~voice = 3;
//freeze [on/off]
~synthInstances[~voice-1].set(\point, 1); //on
~synthInstances[~voice-1].set(\point, 0); //off
*/


/*
*/
//task duration yield
(
    var dur;
    var max = 4;
    var amount = 100;
    dur = Pseq([max, max/2, 1, 0.5, 0.5, 1, 1, max, max/2, max/3], amount).asStream;

    r = Task({
	var delta;
	while {
	    	delta = dur.next;
    		delta.notNil
	} {
    		"Duration: "++delta.postln;
    		~synthInstances[~voice-1].set(\vol, delta);
    		delta.yield;
	}
    }).play(quant: TempoClock.default.beats + 1.0);
)
*/

/*
e = Env([800,300,400,500,200], [1,1,1,1], \exp); // 4 segments
e = Env([0,-1,-0.5,0,1], [1,1,1,1], \sine).plot; // sine
e = Env.new([0,1,0], [0.5, 1]).plot;
*/
//pan control
(
		// reset then play again
		//e = Env([800,300,400,500,200], [1,1,1,1], \exp); // 4 segments
        e = Env([0,-1,-0.5,0,1], [1,1,1,1], \sine);
		~synthInstances[~voice-1].set(\pan, e.asArray);
		//~synthInstances[~voice-1]..set(\t_gate, 1);
)

// granular synth
//~bufferinstances[~voice].bufnum.postln;
~voice = 2;
{
		100.do{ arg j;
			var timeprop = (j/199.0)**3;
			Synth(\sfgrain,[\bufnum, ~bufferinstances[~voice].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1), \pan, 1.0.rand2]);
			rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
		};
}.fork

//task granular trigger
(
    var dur;
    var max = 10;
    var amount = 1;
    dur = Pseq([max, max/2, max/5, 0, 0, max/4, max/3, max, max/2, max/3], amount).asStream;

    r = Task({
	var delta;
	while {
	    	delta = dur.next;
    		delta.notNil
	} {
    		"Duration: "++delta.postln;
    		//~synthInstances[~voice-1].set(\vol, delta);
		{
			100.do{ arg j;
			var timeprop = (j/199.0)**3;
			Synth(\sfgrain,[\bufnum, ~bufferinstances[~voice].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1), \pan, 1.0.rand2]);
			rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
			};
		}.fork;

    		delta.yield;
	}
    }).play(quant: TempoClock.default.beats + 1.0);
)

//pre-perfo
/*
~v1 = "/Users/hordia/samples/freesound-sessions/BREATHE_IN_2.wav";
//TODO: build in one line with a script
~netaddr.sendMsg("/playfreeze", ~v1, 0);
~netaddr.sendMsg("/playfreeze", ~v2, 1);
~netaddr.sendMsg("/playfreeze", ~v3, 2);
~netaddr.sendMsg("/playfreeze", ~v4, 3);
~netaddr.sendMsg("/playfreeze", ~v5, 4);
~netaddr.sendMsg("/playfreeze", ~v6, 5);
~netaddr.sendMsg("/playfreeze", ~v7, 6);
~netaddr.sendMsg("/playfreeze", ~v8, 7);
*/

/*
Voice with explanation about API-Culture and sound from the Cloud...
~mir_voice  = Buffer.read(s, "/Users/hordia/samples/speech-sounds-from-the-cloud.wav");
~voice = Synth(\mutantefreeze, [\bufnum, ~mir_voice.bufnum, \out, 0, \vol, 1])
~voice.set(\point, 0); //off
~voice.set(\point, 1); //on
*/

s.waitForBoot {
	~bufferinstances = Array.newClear(8);
	~synthInstances = Array.newClear(8);

	//freeze
	SynthDef(\mutantefreeze, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startPos=0;
		var in, chain, sig;
		in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
		chain = FFT(LocalBuf(4096), in);
		chain = PV_MagFreeze(chain, point);

		//with PAN
		sig = Splay.ar(vol * IFFT(chain).dup);
		sig = Balance2.ar(sig[0], sig[1], pan);
		Out.ar(out, sig);
	}).add;

	//granular synth / grain synthesis
	SynthDef(\sfgrain, {arg bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04;
		var grain;

		grain= PlayBuf.ar(1,bufnum, BufRateScale.kr(bufnum), 1, BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(0.01,dur),doneAction:2)-0.001);

		Out.ar(0,Pan2.ar(grain, pan))}
	).add;


	//OSC function freeeze synth instance
	x = OSCFunc(
		{ | msg, time, addr, port |
			var new_file, voice_number;

			new_file = msg[1].asString;
			voice_number = msg[2].asInt;
			("Voice number: "+voice_number).postln;
			( "New sound received " + new_file ).postln;

			~bufferinstances[voice_number].free;
			~bufferinstances[voice_number] = Buffer.read(s, new_file);

			~synthInstances[voice_number].free;
			~synthInstances[voice_number] = Synth(\mutantefreeze, [\bufnum, ~bufferinstances[voice_number].bufnum, \out, 0, \vol, 1])
		},

		'/playfreeze'
	);
//x.free;

	////////////////
	//MIDI controller
	////////////////

	~mpkmidimix = AkaiMidiMix(); //también hace el midi connect All

	/////////////////////////////////////
// MIDI out

~default_channel = 0;
~midi_device_set_color = MIDIOut.newByName("MIDI Mix", "MIDI Mix");
~button_light = {
	|button_arg, onoff_arg|
	~midi_device_set_color.noteOn(~default_channel, button_arg, onoff_arg);
};

//Turn off all the buttons light
~mpkmidimix.recpads.do {|pad, i|
	~button_light.value(pad.note, 0); //off
};
~mpkmidimix.mutepads.do {|pad, i|
	~button_light.value(pad.note, 0); //off
};

~button_blink = {
   |button_arg|
	a = Task({
   ~button_light.value(button_arg, 1);
	0.2.wait;
   ~button_light.value(button_arg, 0);
	});
	a.start;
};

/////////////////////////////////////

	//master sound server volume (SuperCollider in DB)
	v = s.volume;
	~mpkmidimix.master[0].onChange = {|val| v.volume = val.linlin(0, 127, v.min, v.max) };
	~maxvol = 4;

	~mpkmidimix.recpads.do {|pad, i|
		pad.onPress = {|val|
			("REC ARM "+(i+1)).postln;
			~tmpsynth = ~synthInstances[i];
			~tmpsynth.get(\point, { arg value;
				if( value >0,{ //on (0 off,  >0 on)
					~tmpsynth.set(\point, 0);
					("MutanteSynth"+(i+1)+": Freeze OFF").postln;
					~button_light.value(pad.note, 0); //off
				}, {
					~tmpsynth.set(\point, 1);
					("MutanteSynth"+(i+1)+": Freeze ON").postln;
					~button_light.value(pad.note, 1); //on
				});
			});
		};
	};



	~granularAmount = [50,50,50,50,50,50,50,50]; //default values
	~granular_voladjust = 10;
	~mpkmidimix.mutepads.do {|pad, i|
		pad.onPress = {|val|
			"Granular synth Pad %\n".postf(i+1);

			~button_blink.value(pad.note);
			{
				('Granular amount '++~granularAmount[i] ).postln;
				~granularAmount[i].do{ arg j;
					var timeprop = (j/199.0)**3;
					Synth(\sfgrain,[\bufnum,  ~bufferinstances[i].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1)+~granular_voladjust, \pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
				};
			}.fork
		};
	};

	~mpkmidimix.knobsA.do {|knob, i|
		//knob.onChange = {|val| ('~granular'++(i+1)++'_amount = '++ val.linlin(0,127,5,500)).interpret };
		knob.onChange = {|val| ~granularAmount[i] = val.linlin(0,127,5,500).asInteger };
	};

	~mpkmidimix.knobsC.do {|knob, i|
		knob.onChange = {|val| ~synthInstances[i].set(\pan, val.linlin(0,127,-1,1))};
	};

	//Sliders --> mutantefreeze_synth volume
	~mpkmidimix.sliders.do {|slider, i|
		slider.onChange = {|val| ~synthInstances[i].set(\vol, val.linlin(0,127,0,~maxvol))};
	};

	//Knobs --> mutantefreeze_synth PAN
	~mpkmidimix.knobsC.do {|knob, i|
		knob.onChange = {|val| ~synthInstances[i].set(\pan, val.linlin(0,127,-1,1))};
	};

} //end of waitForBoot

