/*
    Cloud Instrument & Sounds from the web (using API-Cultor) by hordia

    init & setup
*/

//s.options.sampleRate= 48000; //traktor2 //FIXME: ver soundblaster live! 5.1
s.boot; //start server

s.waitForBoot {
	~bufferinstances = Array.newClear(8);
	~synthInstances = Array.newClear(8);

	//Freeze synth
	~initial_freeze_vol = 0;
	SynthDef(\mutantefreeze, { arg out=0, bufnum=0, point=0, vol=~initial_freeze_vol, fftwidth=4096, pan=0, startPos=0;
				var in, chain, sig;
				in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
				chain = FFT(LocalBuf(4096), in);
				chain = PV_MagFreeze(chain, point);

				//with PAN
				sig = Splay.ar(vol * IFFT(chain).dup);
				sig = Balance2.ar(sig[0], sig[1], pan);
				Out.ar(out, sig);
	}).add;

	//Granular synth / grain synthesis
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
			( "Voice number: "+voice_number).postln;
			( "New sound received " + new_file ).postln;

			~bufferinstances[voice_number].free;
			~bufferinstances[voice_number] = Buffer.read(s, new_file);

			~synthInstances[voice_number].free;
			~synthInstances[voice_number] = Synth(\mutantefreeze, [\bufnum, ~bufferinstances[voice_number].bufnum, \out, 0, \vol, 0])
		},
		'/playfreeze'
	);
//x.free;


//General OSC service functions

//Master sound server volume (SuperCollider in dB)
v = s.volume;
OSCdef.new(
	\fx_volume,
	{
		arg msg;
		//TODO: respuesta audible logarítmica? linexp?
        v.volume = msg[1].asFloat.linlin(0, 10, v.min, v.max);
		("Volume: "++msg[1]).postln;
	},
	'/fx/volume',
);

OSCdef.new(
	\fx_pan,
	{
		arg msg;
		("FIXME // PAN: "++msg[1]).postln;
        //v.pan = msg[1].asFloat;
		Out.ar(0, Pan2(msg[1].asFloat)); //FIXME general PAN
		//Out.ar(1, Pan2(msg[1].asFloat,0));
	},
	'/fx/pan',
);

//test function
OSCdef.new(
	\retrieve,
	{
		arg msg;
			("retrieve: "++msg[1]++", "++msg[2]).postln;

	},
	'/retrieve',
);

//External controller config
"midi_to_osc.scd".loadRelative;
"akai_midimix_to_osc.scd".loadRelative; // AKAI MIDIMix Controller
//"yaeltex_midi_to_osc".loadRelative; // Yaeltex custom MIDI controller
} //end of waitForBoot
