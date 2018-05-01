/*
Cloud Instrument & Sounds from the web (using API-Cultor) by hordia (2018)
<h@ordia.com.ar>

https://github.com/sonidosmutantes/apicultor

    init & setup

1. Run apicultor service:
      apicultor/cloud_instrument$ python CloudInstrument.py
2. Run this SuperCollider script
*/


s.options.sampleRate= 48000; //traktor2 etc //FIXME: ver soundblaster live! 5.1

//TODO: ver multiple outputs
s.boot; //start server

// (optional) Recording
/*
thisProcess.platform.recordingsDir//locate folder
s.recHeaderFormat;//request default format
s.recHeaderFormat = "wav";
//s.recHeaderFormat = "aiff";
s.record;
*/

s.waitForBoot {
	~voices_number = 8; //max simultaneus synth 'voices' (polyphony)
	~bufferinstances = Array.newClear(~voices_number);
	~synthInstances = Array.newClear(~voices_number);

//Mutante main sytnth (Multi effec freeze synth)
	//TODO: check if there is a way to avoid processing when are disabled (pitchratio=1, delamix=0, etc
SynthDef(\mutantefreeze, { arg busout=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startPos=0, delay = 0.25, numchannels=1, delaymix=0, lfoamount= 0, lforate=0, pitchratio=1, roomsize=0.5, reverbmix=0;

	var in, chain, sig, dupsig, adelay, vibrato;

	//plays buffer in loop
	in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
//in = PlayBuf.ar(numchannels, bufnum, BufRateScale.kr(bufnum),loop: 1);

	//with FREEZE (point=0 off)
	chain = FFT(LocalBuf(4096), in);
	chain = PV_MagFreeze(chain, point);
	dupsig = IFFT(chain).dup;

	vibrato = SinOsc.ar(lforate,Rand(0,2.0));
	dupsig = vibrato*dupsig*(lfoamount) + dupsig*(1-lfoamount);

	//with DELAY
	adelay = DelayN.ar( dupsig, delay, delay, delaymix, dupsig); // input is mixed with delay via the add input (delaymix = 0 -> off)

	//with PAN
	sig = Splay.ar(vol *adelay);
	sig = Balance2.ar(sig[0], sig[1], pan);

	//with Pitch Shift (rate)
		//pitchRatio - the ratio of the pitch shift. Must be from 0.0 to 4.0.
	sig = PitchShift.ar(sig, 0.1, pitchratio, 0, 0.004);

	//with Reverb (in, mix: 0.33, room: 0.5, damp: 0.5, mul: 1, add: 0)
	sig = FreeVerb.ar(sig, reverbmix, roomsize, 0.5, 1, 0 );
	//sig = GVerb(sig, roomsize, 3, 0.5, 0.5, 15 );

	//Filter
		//x = {LPF.ar(WhiteNoise.ar)}.play(s);

	//Other FXs
	//right = FreqShift.ar(left, SinOsc.kr(3.23, 0, 5));

	Out.ar(busout, sig);
}).add;

/*
	//Testing. Reloading first voice
	~synthInstances[0].free;
	~synthInstances[0] = Synth(\mutantefreeze, [\bufnum, ~bufferinstances[0].bufnum, \out, 0, \vol, 1]);
*/

//Granular synth / grain synthesis
SynthDef(\sfgrain, {arg bufnum=0, pan=0.0, startPos=0.0, amp=0.1, dur=0.04;
	var grain;

	grain= PlayBuf.ar(1,bufnum, BufRateScale.kr(bufnum), 1,
			BufFrames.ir(bufnum)*startPos, 0)*(EnvGen.kr(Env.perc(0.01,dur),doneAction:2)-0.001);

	Out.ar(0,Pan2.ar(grain, pan))}
).add;



/////////////////////////////////
//External controller config
//WARNING: se rompe si no esta el controlador presente/conectado
//"ui/akai_midimix_to_osc.scd".loadRelative; // AKAI MIDIMix Controller
"ui/yaeltex_midi_to_osc.scd".loadRelative; // Yaeltex custom MIDI controller
/////////////////////////////////


//
//OSC function freeeze synth instance
// Recibe de apicultor la notificación del que nuevo sonido esta disponible
//
x = OSCFunc(
		{ | msg, time, addr, port |
			var new_file, voice_number, metadata;

			new_file = msg[1].asString;
			voice_number = msg[2].asInt;
			( "Voice number: "+(voice_number+1)).postln;
			( "New sound received " + new_file ).postln;

			metadata = msg[3].asString; //Ej: sinbase.wav by anbo - id: 18860

			~bufferinstances[voice_number].free;
			~bufferinstances[voice_number] = Buffer.read(s, new_file);

			~synthInstances[voice_number].free;

			//~default_volume = 0;
			~default_volume = 1;
			~synthInstances[voice_number] = Synth(\mutantefreeze, [\bufnum, ~bufferinstances[voice_number].bufnum, \out, 0, \vol, ~default_volume]);

			//Sets metadata in OSC display
			~osc_server_OpenStage.sendMsg('/EDIT', 'soundfile', '{"value":"'++metadata++'","color":"green"}');
			metadata = metadata.split($ ).postln; //convert it to an string array, first position is the filename
			~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(voice_number+1), '{"label":"'++metadata[0]++'","color":"green"}'); //sets wav filename
			~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(voice_number+1), '{"value":"1"}'); //sets led on
		},
		'/playnewsound'
);
//x.free;

//Error searching/retrieving a new soundFile
/*
	        Possible errors:
        * Internet Connection
        * Database error
        * No file available with those descriptors values
        * File format or convertion error
        * No available space error
        * Etc
*/
OSCdef.new(
	\error_with_the_new_sound,
	{
		arg msg;
		var metadata_error = msg[1].asString;

		(metadata_error).postln;

		//custom error message
		metadata_error = "Error en la descarga. Probar otra descripcion sonora";
		~osc_server_OpenStage.sendMsg('/EDIT', 'soundfile', '{"value":"'++metadata_error++'","color":"red"}');

	},
	'/errorwithnewsound',
);

OSCdef.new(
	\chat_msg,
	{
		arg msg;
		var chat = msg[1].asString;
		(chat).postln;
	},
	'/chat',
);


//General OSC service functions

//Master sound server volume (SuperCollider in dB)
//Supported commands

//Master Volume
v = s.volume;
OSCdef.new(
	\fx_volume,
	{
		arg msg;
		//TODO: respuesta audible logarítmica? linexp?
        v.volume = msg[1].asFloat.linlin(0, 10, v.min, v.max);
		("Volume: "++msg[1].asFloat.asStringPrec(2)).postln;
	},
	'/fx/volume',
);

//Set Voice
~active_voice = 1; //1..8
OSCdef.new(
	\set_voices,
	{
		arg msg;
		if( msg[1].asInteger>0, {
			~active_voice = msg[1].asInteger;
			("Setting voice: "++~active_voice).postln;


			//OpenStage UI update
			[1,2,3,4,5,6,7,8].do{|number|
				~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(number), '{"value":"0","color":"green"}');
			};
			~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(~active_voice), '{"value":"1","color":"green"}');
		 });
	},
	'/set_voices',
);

//Retrieve
OSCdef.new(
	\retrieve,
	{
		arg msg;
			("Ask to retrieve a new sound from database").postln;
			//msg[1] argument, always 1
			~onoff_value = 1;
			//Activating ~active_voice on apicultor
			~osc_server_APICultorService.sendMsg("/set_voices", ~active_voice, ~onoff_value);
			~osc_server_APICultorService.sendMsg("/retrieve", 1); //get a new sound
			//~send_osc_all.value(~new_sound_reset, 1); // RESET MIR config?


			//OpenStage UI update
			[1,2,3,4,5,6,7,8].do{|number|
				//sets led off
				~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(number), '{"value":"0"}');
			};
	},
	'/retrieve',
);

//Reset MIR state
OSCdef.new(
	\reset_mir,
	{
		arg msg;
			("Reset MIR state! descriptors_dict={} clean").postln;
			~osc_server_APICultorService.sendMsg("/mir/reset", 1); //reset mir state
	},
	'/mir/reset',
);

//Mutante Granular SYNTH
~granularAmount = [50,50,50,50,50,50,50,50]; //default values for each voice (1..8)
~granular_voladjust = 10;
OSCdef.new(
	\synth_granular_trigger,
	{
		arg msg;
		var v_pos = ~active_voice-1;
			("Granular synthesis of voice "++(~active_voice)++' (value '++msg[1].asInteger++')').postln;
			('-> Grains amount '++~granularAmount[v_pos] ).postln;

		if( msg[1].asInteger==1, 	{
			{
			~granularAmount[v_pos].do{ arg j;
					var timeprop = (j/199.0)**3;

					//TODO: save reference to adjust volume
					Synth(\sfgrain,[\bufnum,  ~bufferinstances[v_pos].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1)+~granular_voladjust, \pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
				};
			}.fork
		});

	},
	'/synth/granular/trigger',
);

//Grains amount update (for each voice)
OSCdef.new(
	\synth_granular_set_grains,
	{
		arg msg;
		var v_pos = ~active_voice-1;
		~granularAmount[v_pos]  = msg[1].asInteger;

		("Grain amount "++~granularAmount[v_pos] ++' (voice '++~active_voice++')').postln;
	},
	'/synth/granular/grains/amount',
);

//Freeze synth (enable/disable) with 1 freeze sets on, with 0 sets off
OSCdef.new(
	\synth_freeze,
	{
		arg msg;
		var value = msg[1].asInteger;
		var voice = msg[2].asFloat;

			if( voice==(-1), {
				voice = ~active_voice-1;
			});

			//("Value: "++value).postln;
		    ~synthInstances[voice].set(\point, value); //1 freeze sets on, with 0 sets off
		    if( msg[1].asInteger==1,
				{("Freeze ON (voice "++(voice+1)++")").postln;},
				{("Freeze OFF (voice "++(voice+1)++")").postln;},
            );
	},
	'/synth/freeze',
);

//////////////////////////////////
//(single) Voice configuration (FX)
//////////////////////////////////

//Voice volume
OSCdef.new(
	\voice_volume,
	{
		arg msg;
			("Voice: "++(~active_voice)++" , Vol:  "++msg[1].asFloat).postln;
			~synthInstances[~active_voice-1].set(\vol, msg[1].asFloat );
	},
	'/voice_volume',
);

//Voice PAN
OSCdef.new(
	\fx_pan,
	{
		arg msg;
		var value = msg[1].asFloat;
			 //FIXME general PAN
			//("PAN: "++msg[1]).postln;
			//Out.ar(0, Pan2(value));
			//Out.ar(1, Pan2(msg[1].asFloat,0));

		("Voice: "++(~active_voice)++" , Pan:  "++value.asStringPrec(2)).postln;
		~synthInstances[~active_voice-1].set(\pan, value );
	},
	'/fx/pan',
);

//Delay amount. With delay==0, sets delaymix to 0 (no delay, dry signal)
OSCdef.new(
	\fx_delay,
	{
		arg msg;
			("Voice: "++(~active_voice)++" , Delay:  "++msg[1].asFloat.asStringPrec(2)).postln;
			if( msg[1].asFloat==0,
			   {
					~synthInstances[~active_voice-1].set(\delaymix,0);
					"Delaymix OFF".postln

			   },
			   ~synthInstances[~active_voice-1].set(\delaymix, ~default_delaymix),
	     	);
			~synthInstances[~active_voice-1].set(\delay, msg[1].asFloat );
	},
	'/fx/delay',
);

//Delaymix, amount of wet/dry signal
OSCdef.new(
	\fx_delaymix,
	{
		arg msg;
			("Voice: "++(~active_voice)++" , DelayMix:  "++msg[1].asFloat.asStringPrec(2)).postln;
			~synthInstances[~active_voice-1].set(\delaymix, msg[1].asFloat);
	},
	'/fx/delay/mix',
);


//(pitch) Rate
OSCdef.new(
	\fx_pitch_rate,
	{
		arg msg;
			("Voice: "++(~active_voice)++" , (pitch) Rate:  "++msg[1].asFloat).postln;
			~synthInstances[~active_voice-1].set(\pitchratio, msg[1].asFloat);
	},
	'/fx/rate',
);


//Gate. TODO: improve this
OSCdef.new(
	\fx_gate,
	{
		arg msg;
			var value = msg[1].asFloat;
			if( value>0, {value=1}, {value=0});
			("Voice: "++(~active_voice)++" , Gate:  "++value).postln;
			~synthInstances[~active_voice-1].set(\vol, value);
	},
	'/fx/gate',
);


//Reverb
OSCdef.new(
	\fx_reverb,
	{
		arg msg;
			var value = msg[1].asFloat;
			("Voice: "++(~active_voice)++" , Reverb:  "++value.asStringPrec(2)).postln;
			~synthInstances[~active_voice-1].set(\reverbmix, value);
	},
	'/fx/reverb',
);

//Ressonance (by now reverb roomsize)
OSCdef.new(
	\fx_ressonance,
	{
		arg msg;
			var value = msg[1].asFloat;
			("Voice: "++(~active_voice)++" , Ressonance (reverb roomsize):  "++value.asStringPrec(2)).postln;
			~synthInstances[~active_voice-1].set(\roomsize, value.linlin(0,1,0,4));
	},
	'/fx/ressonance',
);

//TODO: implement
//Filter (LP/BP/HP) -> 0,1,2
OSCdef.new(
	\fx_filter,
	{
		arg msg;
			var value = msg[1].asFloat;
			//if( value>0, {value=1}, {value=0});
			("(TODO: implement) Voice: "++(~active_voice)++" , Filter Mode:  "++value.asStringPrec(2)).postln;
			//~synthInstances[~active_voice-1].set(\lpf, value);
			//~synthInstances[~active_voice-1].set(\bpf, value);
			//~synthInstances[~active_voice-1].set(\hpf, value);
	},
	'/fx/filter',
);

//TODO: implement
//Cutoff
OSCdef.new(
	\fx_cutoff,
	{
		arg msg;
			var value = msg[1].asFloat;
			("(TODO: implement) Voice: "++(~active_voice)++" , Cutoff,:  "++value.asStringPrec(2)).postln;
			//~synthInstances[~active_voice-1].set(\filtercutoff,, value);
	},
	'/fx/cutoff',
);

//LFO (By now is RATE, not amount)
OSCdef.new(
	\fx_lfo,
	{
		arg msg;
			("Voice: "++(~active_voice)++" , LFO (rate):  "++msg[1].asFloat.asStringPrec(2)).postln;
			~synthInstances[~active_voice-1].set(\lforate, msg[1].asFloat);
	},
	'/fx/lfo',
);

//Factor --> LFO amount mix
OSCdef.new(
	\fx_factor,
	{
		arg msg;
			var value = msg[1].asFloat;
			("Voice: "++(~active_voice)++" , Rate (LFO amount mix):  "++msg[1].asFloat.asStringPrec(2)).postln;
			~synthInstances[~active_voice-1].set(\lfoamount, msg[1].asFloat.linlin(0,1,0,0.95));
	},
	'/fx/factor',
);

~osc_server_APICultorService.sendMsg("/chat", "Hello. SuperCollider is ready to rock!");

//Startup sounds
} //end of waitForBoot
