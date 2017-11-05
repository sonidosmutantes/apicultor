//Address and port
//NetAddr.localAddr //check '127.0.0.1', 57120

//s.options.sampleRate= 48000;
s.boot; //start server

s.waitForBoot {

	~bufferinstances = [];
	~synthInstances = [];

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


	//OSC function freeeze synth instance
	x = OSCFunc(
		{ | msg, time, addr, port |
			var new_file, voice_number;

			new_file = msg[1].asString;
			voice_number = msg[2].asString;
			~bufferinstances.add(  Buffer.read(s, new_file) );
			("Voice number: "+voice_number).postln;
			( "New sound received " + new_file ).postln;

			~synthInstances.add(Synth(\mutantefreeze, [\bufnum, b.bufnum, \out, 0, \vol, 1]))
		},

		'/playfreeze'
	);

	~mpkmidimix = AkaiMidiMix(); //también hace el midi connect All

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
	~granular_voladjust = 4;
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
} //end of waitForBoot


