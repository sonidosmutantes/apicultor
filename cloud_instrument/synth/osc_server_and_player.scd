//Address and port
//NetAddr.localAddr //check '127.0.0.1', 57120

//s.options.sampleRate= 48000;
s.boot; //start server

s.waitForBoot {

	//create the buffer
	b = Buffer.alloc(s, s.sampleRate * 8.0, 2); // an 8 second stereo buffer

	//play
	SynthDef(\playBufMono,
		{| out = 0, bufnum = 0, vol=1, rate = 1 |

			var scaledRate, player;
			scaledRate = rate * BufRateScale.kr(bufnum);
			player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);
			Out.ar(out, vol * player).dup }
	).add;

	//freeze
	SynthDef(\mutantefreeze, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startPos=0;
		var in, chain, sig;
		in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
		chain = FFT(LocalBuf(4096), in);
		chain = PV_MagFreeze(chain, point);

		//Out.ar(out, vol * IFFT(chain).dup);

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


	//Playing resource
	x = OSCFunc(
		{ | msg, time, addr, port |
			var pyFile;

			~new_file = msg[1].asString;
			~voice_number = msg[2].asString;
			b = Buffer.read(s, ~new_file);
			("Voice number: "+~voice_number).postln;
			( "New sound received " + ~new_file ).postln;

			l = Synth(\playBufMono, [\out, 0, \bufnum, b.bufnum, \rate, 1]);
			r = Synth(\playBufMono, [\out, 1, \bufnum, b.bufnum, \rate, 1]);
		},

		'/play'
	);

	x = OSCFunc(
		{ | msg, time, addr, port |
			var pyFile;

			~new_file = msg[1].asString;
			~voice_number = msg[2].asString;
			b = Buffer.read(s, ~new_file);
			("Voice number: "+~voice_number).postln;
			( "New sound received " + ~new_file ).postln;

			~tmp_mutante_freeze = Synth(\mutantefreeze, [\bufnum, b.bufnum, \out, 0, \vol, 1]);
		},

		'/playfreeze'
	);

	~mpkmidimix = AkaiMidiMix(); //también hace el midi connect All

	//Freeze control first button
	~mpkmidimix.recpads[0].onPress = {|val|
		("REC ARM 1").postln;
		~tmpsynth = ~tmp_mutante_freeze;
		~tmpsynth.get(\point, { arg value;
			if( value >0,{ //on (0 off,  >0 on)
				~tmpsynth.set(\point, 0);
				("MutanteSynth: Freeze OFF").postln;
				~button_light.value(~mpkmidimix.recpads[0].note, 0); //off
			}, {
				~tmpsynth.set(\point, 1);
				("MutanteSynth: Freeze ON").postln;
				~button_light.value(~mpkmidimix.recpads[0].note, 1); //on
			});
		});
	};

	//master sound server volume (SuperCollider in DB)
	v = s.volume;
	~mpkmidimix.master[0].onChange = {|val| v.volume = val.linlin(0, 127, v.min, v.max) };


	~maxvol = 4;
	~mpkmidimix.sliders[0].onChange = {|val| ~tmp_mutante_freeze.set(\vol, val.linlin(0,127,0,~maxvol))};

	~granularAmount = 200;
	~mpkmidimix.mutepads[0].onPress = {|val|
			"Granular synth Pad %\n".postf(1);

			~button_blink.value(~mpkmidimix.mutepads[0].note);
			{
				('Granular amount '++~granularAmount ).postln;
				~granularAmount.do{ arg j;
					var timeprop = (j/199.0)**3;
					Synth(\sfgrain,[\bufnum,  b.bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1), \pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
				};
			}.fork
	};


} //end of waitForBoot


