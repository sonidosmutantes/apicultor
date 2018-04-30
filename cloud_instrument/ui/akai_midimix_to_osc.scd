///////////////////////////////////////////////////////////////////////////
// Akai MIDI mix
///////////////////////////////////////////////////////////////////////////
"midi_to_osc.scd".loadRelative;
~mpkmidimix = AkaiMidiMix(); //también hace el midi connect All

//WARNING: en raspberry pi figura como El midi out "API-cultor MIDI 1"

// Test messages
//~send_osc_all.value("/fx/volume", 1.1); //test volume msg
//~send_osc_all.value("/fx/pan", -1); //test pan msg

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


//BANK LEFT -> MIR mode
~mode = 0; //0 es MIR , 1 es SYNTH
//set up default modefa
        ~set_mode_mir.value;
		~button_light.value(~mpkmidimix.bankLeft[0].note, 1); //led on
		~button_light.value(~mpkmidimix.bankRight[0].note, 0); //led off

~mpkmidimix.bankLeft[0].onPress = {|val|
		//("BANK LEFT ").postln;
				// ("BANK RIGHT ").postln;
			~osc_server_SuperCollider.sendMsg(~mir_mode, 1);
			~osc_server_SuperCollider.sendMsg(~synth_mode, 0);
	    //~send_osc_all.value(~mir_mode, 1);
	    //~send_osc_all.value(~synth_mode, 0);
		("MIR Mode ON").postln;
		~mode = 0;
	//~set_mode_mir.value; //its called in osc receiver
		~button_light.value(~mpkmidimix.bankLeft[0].note, 1); //led on
		~button_light.value(~mpkmidimix.bankRight[0].note, 0); //led off
};

//BANK RIGHT -> Synth mode
~mpkmidimix.bankRight[0].onPress = {|val|
		// ("BANK RIGHT ").postln;
			~osc_server_SuperCollider.sendMsg(~mir_mode, 0);
			~osc_server_SuperCollider.sendMsg(~synth_mode, 1);
	    //~send_osc_all.value(~mir_mode, 0);
	    //~send_osc_all.value(~synth_mode, 1);
		("SYNTH Mode ON").postln;
		~mode = 1;
        // ~set_mode_synth.value; //its called in osc receiver
		~button_light.value(~mpkmidimix.bankRight[0].note, 1); //led on
		~button_light.value(~mpkmidimix.bankLeft[0].note, 0); //led off
};

//SOLO (master) button mapped to '/retrieve' new sound
~mpkmidimix.solo[0].onPress = {|val|
	//("SOLO btn to retrieve new sound").postln;
	~osc_server_SuperCollider.sendMsg(~new_sound_retrieve, 1); //sends 1


	//Blink many buttons (visual feedback)
	//TODO: revisar si tiene (o funciona?) el led del SOLO button
	~button_blink.value(~mpkmidimix.solo[0].note); //blink
	~mpkmidimix.mutepads.do {|pad, i| ~button_blink.value(pad.note)}; //led on

/*
	//Joystick OpenStage movement
	a = Task({
	       ~osc_server_OpenStage.sendMsg(~new_sound_retrieve, 0.55, 1); //UP
	        0.2.wait;
			~osc_server_OpenStage.sendMsg(~new_sound_retrieve, 0.55, 0); //Down
	        0.2.wait;
           ~osc_server_OpenStage.sendMsg(~new_sound_retrieve, 0.55, 0.55); //CENTER
	});
	a.start;
*/

	//TODO: agregar que se apague la luz cuando el service de apicultor devuelve el sonido
	//solo encender en esta etapa/función y apagar al recibir otro mensaje...
	//~button_light.value(~mpkmidimix.solo[0].note, 1); //led on
	//~button_light.value(~mpkmidimix.solo[0].note, 0); //led off
};

/////////////////////////////////////
// Row 1 (knobs)
/////////////////////////////////////

//LFO Amount
~mpkmidimix.knobsA[0].onChange = {|val| ("TODO: add osc function LFO amount "++val.linlin(0,127,0,1)).postln};

//Rate
~mpkmidimix.knobsA[1].onChange = {|val| ("TODO: add osc function Rate "++val.linlin(0,127,0,1)).postln};


//BPM
~mpkmidimix.knobsA[4].onChange = {|val|
	val = val.linlin(0,127,40,200).asFloat;
	("MIR: BPM "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_bpm, val);
};

//Key
~mpkmidimix.knobsA[5].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Key (now chord strength) "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_key, val);
};


//Master sound server PAN
~mpkmidimix.knobsA[6].onChange = {|val| ~send_osc_all.value(~fx_pan, val.linlin(0,127,-1,1))};

//Master sound server VOLUME
~mpkmidimix.knobsA[7].onChange = {|val| ~send_osc_all.value(~fx_volume, val.linlin(0,127,0,10))};

/////////////////////////////////////
// Row 2 (knobs)
/////////////////////////////////////

//Synth
//Gate
~mpkmidimix.knobsB[0].onChange = {|val| ("TODO: add osc function Gate amount "++val.linlin(0,127,0.4,1.4).asInteger).postln};

//Reverb
~mpkmidimix.knobsB[1].onChange = {|val| ("TODO: add osc function Reverb amount "++val.linlin(0,127,0,1).asFloat).postln};

//Delay
~mpkmidimix.knobsB[2].onChange = { |val|
	val = val.linlin(0,127,0,~delay_max).asFloat;
	("SYNTH: Delay "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_delay, val);
};

//Tone
~mpkmidimix.knobsB[3].onChange = { |val|
	val = val.linlin(0,127,0,1).asFloat;
	("SYNTH: Tone (delaymix by now) "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_delaymix, val);
};


//MIR - Note: add supported MIR descriptors in MIRState.py
//Duration
~mpkmidimix.knobsB[4].onChange = {|val|
	val = val.linlin(0,127,0.1,5).asFloat; //TODO: define range
	("MIR: Duration "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_duration, val);
};

//Pitch
//	Searchs for "pitch.mean": "=", #TODO: add a range of tolerance
~mpkmidimix.knobsB[5].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Pitch "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_pitch, val);
};

//Inharmonicity
~mpkmidimix.knobsB[6].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Inharmonicity "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_inharmonicity, val);
};

//Dissonance
~mpkmidimix.knobsB[7].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Dissonance "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_dissonance, val);
};

/////////////////////////////////////
// Row 3 (knobs)
/////////////////////////////////////

//Synth
//LP/BP/HP Filter
~mpkmidimix.knobsC[0].onChange = {|val| ("SYNTH: TODO: add osc function LP/BP/HP Filter amount "++val.linlin(0,127,0,1).asFloat).postln};

//Cutoff
~mpkmidimix.knobsC[1].onChange = {|val| ("SYNTH: TODO: add osc function Cutoff amount "++val.linlin(0,127,0,1).asFloat).postln};

//Ressonance
~mpkmidimix.knobsC[2].onChange = {|val| ("SYNTH: TODO: add osc function Ressonance amount "++val.linlin(0,127,0,1).asFloat).postln};

//Factor
~mpkmidimix.knobsC[3].onChange = {|val| ("SYNTH: TODO: add osc function Factor amount "++val.linlin(0,127,0,1).asFloat).postln};

//MIR
//HFC
~mpkmidimix.knobsC[4].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: HFC "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_hfc, val);
};

//Pitch Salience
~mpkmidimix.knobsC[5].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: PitchSalience "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_pitch_salience, val);
};

//SpectralCentroid
~mpkmidimix.knobsC[6].onChange = {|val|
	val = val.linlin(0,127,0,10000).asFloat;
	("MIR: SpectralCentroid "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_spectral_centroid, val);
};

//Spectral Complexity
~mpkmidimix.knobsC[7].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: SpectralComplexity "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_spectral_complexity, val);
};



/////////////////////////////////////
// Row 5 (sliders)
/////////////////////////////////////

///////////////////////////////////////
//Voices (first 4 mute and rec buttons)
///////////////////////////////////////
//Voices 1 to 4
~mpkmidimix.mutepads[0..3].do {|pad, i|
	pad.onPress = {
		~osc_server_SuperCollider.sendMsg(~set_voices, i+1);

		//midimix conf:
		~mpkmidimix.mutepads[0..3].do {|pad, i|
			~button_light.value(pad.note, 0); //off
		};
		~mpkmidimix.recpads[0..3].do {|pad, i|
		~button_light.value(pad.note, 0); //off
		};
		~button_light.value(pad.note, 1);
	};
};

//Voices 5 to 8
~mpkmidimix.recpads[0..3].do {|pad, i|
	pad.onPress = {
		~osc_server_SuperCollider.sendMsg(~set_voices, i+5);

		//midimix conf:
		~mpkmidimix.mutepads[0..3].do {|pad, i|
			~button_light.value(pad.note, 0); //off
		};
		~mpkmidimix.recpads[0..3].do {|pad, i|
		~button_light.value(pad.note, 0); //off
		};
		~button_light.value(pad.note, 1);
	};
};

//Mute btn 7
~mpkmidimix.mutepads[7].onPress = {
	~osc_server_SuperCollider.sendMsg(~mir_reset, 1);
	~button_blink.value(~mpkmidimix.mutepads[7].note);
}

//RecArm5:: Triggers grain synth
~mpkmidimix.recpads[4].onPress = {
	~osc_server_SuperCollider.sendMsg("/synth/granular/trigger", 1);
	~button_blink.value(~mpkmidimix.recpads[4].note);
};

//Slider 2 (sets volume)
~mpkmidimix.sliders[1].onChange = {|val|
	~osc_server_SuperCollider.sendMsg("/voice_volume", val.linlin(0,127,0,~max_voice_volume) );
};

//Slider 4 (enables freeze synth)
~mpkmidimix.sliders[3].onChange = {|val|
	~osc_server_SuperCollider.sendMsg("/synth/freeze", val.linlin(0,127,0,1), -1 ); //-1 means active_voice
};

//Slider 5 (sets grains amount)
~mpkmidimix.sliders[4].onChange = {|val|
	~osc_server_SuperCollider.sendMsg("/synth/granular/grains/amount", val.linlin(0,127,0,~max_grains_amount) );
};

//Master sound server volume (AGAIN, double definition)
~mpkmidimix.master[0].onChange = {|val|
    ~osc_server_SuperCollider.sendMsg(~fx_volume, val.linlin(0,127,0,10))
};

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
//OLD
/////////////////////////////////////////////////////////////////////////////////////
/*
//PAN
~mpkmidimix.knobsC.do {|knob, i|
	//each freeze synth knob
	knob.onChange = {|val| ~synthInstances[i].set(\pan, val.linlin(0,127,-1,1))};
	knob.onChange = {|val| ~send_osc_all.value(~fx_pan, val.linlin(0,127,-1,1))}; //General PAN
};
*/


/*

//Mutante FREEZE synth ON/OFF
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

	//Sliders --> mutantefreeze_synth volume
	~mpkmidimix.sliders.do {|slider, i|
		slider.onChange = {|val| ~synthInstances[i].set(\vol, val.linlin(0,127,0,~maxvol))};
	};

//Mutante Granular SYNTH
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


*/

/*
~send_osc_push_on_button = {
	|a_arg, b_arg, url_arg|

    a_arg.sendMsg(url_arg, 1); //on
    b_arg.sendMsg(url_arg, 1); //on

};

~send_osc_toggle_button = {
	//push button (not toggle) sends 1 value waits 0.3 seconds and send 0 value
	|a_arg, b_arg, url_arg|

    a_arg.sendMsg(url_arg, 1); //on
    b_arg.sendMsg(url_arg, 1); //on
    0.3.wait;
	a_arg.sendMsg(url_arg, 0); //off
	b_arg.sendMsg(url_arg, 0); //off
};



/*
 MIDI to OSC mapping
 AKAI MIDIMIX
 8 canales, de izq a derecha (slides)
 TODO: hacer script python que arme todo esto!
*/
~cc_col1_knob1 = 16;
~cc_col1_knob2 = 17;
~cc_col1_knob3 = 18;
~cc_slider1 = 19;

~cc_col2_knob1 = 20;
~cc_col2_knob2 = 21;
~cc_col2_knob3 = 22;
~cc_slider2= 23;

~cc_col3_knob1 = 24;
~cc_col3_knob2 = 25;
~cc_col3_knob3 = 26;
~cc_slider3= 27;

~cc_col4_knob1 = 28;
~cc_col4_knob2 = 29;
~cc_col4_knob3 = 30;
~cc_slider4= 31;

~cc_col5_knob1 = 46;
~cc_col5_knob2 = 47;
~cc_col5_knob3 = 48;
~cc_slider5= 49;

~cc_col6_knob1 = 50;
~cc_col6_knob2 = 51;
~cc_col6_knob3 = 52;
~cc_slider6= 53;

~cc_col7_knob1 = 54;
~cc_col7_knob2 = 55;
~cc_col7_knob3 = 56;
~cc_slider7= 57;

~cc_col8_knob1 = 58;
~cc_col8_knob2 = 59;
~cc_col8_knob3 = 60;
~cc_slider8= 61;

~cc_slider_master = 62;



MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln; //monitor
	switch(num,
		//fx
		~cc_slider_master,{~send_osc.value(o, p, ~fx_volume, val.linlin(0,127,0,10))}, //TODO: respuesta audible logarítmica? linexp?
		~cc_col8_knob3,{~send_osc.value(o, p, ~fx_pan, val.linlin(0,127,-1,1))},

		//MIR descriptors
		~cc_col1_knob3,{~send_osc.value(o, p,~mir_hfc, val.linlin(0,127,0,1))},
		~cc_col2_knob3,{~send_osc.value(o, p,~mir_spectral_complexity, val.linlin(0,127,0,1))},
		~cc_col3_knob3,{~send_osc.value(o, p,~mir_inharmonicity, val.linlin(0,127,0,1))},
		~cc_col4_knob3,{~send_osc.value(o, p,~mir_pitch_centroid, val.linlin(0,127,0,1))},

		~cc_slider1,{~send_osc.value(o, p,~mir_spectral_centroid, val.linlin(0,127,0,10000))},
        ~cc_slider2,{~send_osc.value(o, p,~mir_duration, val.linlin(0,127,0,15))},
		~cc_slider3,{~send_osc.value(o, p,~mir_bpm, val.linlin(0,127,0,200))},
	)
};

~noteon_bank_left = 25;
~noteon_bank_right = 26;
~noteon_solo_master = 27;

// solo buttons
~noteon_solo_1 = 1;
~noteon_solo_2 = 4;
~noteon_solo_3 = 7;
~noteon_solo_4 = 10;
~noteon_solo_5 = 13;
~noteon_solo_6 = 16;
~noteon_solo_7 = 19;
~noteon_solo_8 = 22;

~noteon_recarm_1 = 3;
~noteon_recarm_2 = 6;
~noteon_recarm_3 = 9;
~noteon_recarm_4 = 12;
~noteon_recarm_5 = 15;
~noteon_recarm_6 = 18;
~noteon_recarm_7 = 21;
~noteon_recarm_8 = 24;

/*
//Enabled Buttons (deprecated) ON/OFF each MIR descriptor
~mir_spectral_centroid_enabled = "/mir/spectral_centroid/mean/enabled";
~mir_duration_enabled = "/mir/duration/enabled";
~mir_bpm_enabled = "/mir/bpm/enabled";

~mir_inharmonicity_enabled = "/mir/inharmonicity/mean/enabled";
~mir_hfc_enabled = "/mir/hfc/mean/enabled";
~mir_spectral_complexity_enabled = "/mir/spectral_complexity/mean/enabled";
~mir_pitch_centroid_enabled = "/mir/pitch_centroid/mean/enabled";
*/
//button



MIDIFunc.noteOn({ |veloc, num, chan, src|
	[chan,num,veloc].postln; //monitor
	switch(num,

		//solo (master) button mapped to /retrieve new sound
		~noteon_solo_master,{
			~send_osc.value(o, p, ~new_sound_retrieve, 1); //sends 1
			~send_osc_toggle_button.value(o,p);
			"solo btn to retrieve new sound".postln;
		},

		~noteon_bank_left,{
			//MIR mode
			~send_osc.value(o, p,~synth_mode, 0);
			~send_osc.value(o, p,~mir_mode, 1);
			"MIR mode".postln;
		},
		~noteon_bank_right,{
			//Synth mode
			~send_osc.value(o, p,~synth_mode, 1);
			~send_osc.value(o, p,~mir_mode, 0);
			"Synth mode".postln;
		},

	    ~noteon_solo_8,{ //last mute button
			~send_osc.value(o, p,~new_sound_reset, 1);
		},

	)
});
*/

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
