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
if( ~host=="mac", {
	~midi_device_set_color = MIDIOut.newByName("MIDI Mix", "MIDI Mix"); //mac
}
,{
	~midi_device_set_color = MIDIOut.newByName("MIDI Mix", "MIDI Mix MIDI 1"); //raspberry pi
});

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
//Note: FX's sends message to osc_server_SuperCollider and MIR sends to osc_server_APICultorService

//LFO (rate, not amount)
~mpkmidimix.knobsA[0].onChange = {|val|
	val = val.linlin(0,127,0,100).asFloat;
	("SYNTH: LFO amount "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_lfo, val);
};

//Rate (Pitch is 0..4 range)
~mpkmidimix.knobsA[1].onChange = {|val|
		val = val.linlin(0,127,0,4).asFloat;
	("SYNTH: Rate "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_rate, val);
};


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
~mpkmidimix.knobsA[6].onChange = {|val| ~osc_server_SuperCollider.sendMsg(~fx_pan, val.linlin(0,127,-1,1))};

//Master sound server VOLUME
~mpkmidimix.knobsA[7].onChange = {|val|
	~osc_server_SuperCollider.sendMsg(~fx_volume, val.linlin(0,127,0,~master_volume_max).asFloat;);
};



/////////////////////////////////////
// Row 2 (knobs)
/////////////////////////////////////

//Synth
//Gate
~mpkmidimix.knobsB[0].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("SYNTH: Gate "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_gate, val);
};

//Reverb
~mpkmidimix.knobsB[1].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("SYNTH: Reverb "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_reverb, val);
};

//Delay
~mpkmidimix.knobsB[2].onChange = { |val|
	val = val.linlin(0,127,0,~delay_max).asFloat;
	("SYNTH: Delay "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_delay, val);
};

//Tone (temporary delaymix)
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
//Filter (LP/BP/HP) -> 0-> OFF // 1:LPF,2:BPF,3:HPF
~mpkmidimix.knobsC[0].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	if( val<0.1, { val=0 }, { if( val<0.38, {val=1},{ if( val<0.68, {val=2}, {val=3})})});
	("SYNTH: Filter Mode "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_filter, val);
};

//Cutoff
~mpkmidimix.knobsC[1].onChange = {|val|
	val = val.linlin(0,127,20,15000).asFloat;
	("SYNTH: Cutoff "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_cutoff, val);
};

//Ressonance
~mpkmidimix.knobsC[2].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("SYNTH: Ressonance "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_ressonance, val);
};

//Factor
~mpkmidimix.knobsC[3].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("SYNTH: Factor "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_factor, val);
};

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
};

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
	~osc_server_SuperCollider.sendMsg(~fx_volume, val.linlin(0,127,0,~master_volume_max).asFloat;);
};

/*
 Documentation
 MIDI to OSC mapping
 AKAI MIDIMIX
 8 channels, from left to right (slides)

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
*/


