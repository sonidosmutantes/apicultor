//MIDIIn.connectAll; //"MIDIEndPoint("API-cultor", "API-cultor")"
//0.5.wait;
"midi_to_osc.scd".loadRelative;
~yaeltex = YaeltexAPICultor(); //también hace el midi connect All

//TODO: detectar donde corre. EN mac  MIDIOut.newByName("API-cultor", "API-cultor")
// y en rapsberry pi MIDIOut.newByName("API-cultor", "API-cultor MIDI 1"); //raspberry pi

///////////////////////////////////////////////////////////////////////////
// Yaeltex custom MIDI controller
///////////////////////////////////////////////////////////////////////////

// First row of 4 knobs (into the right)
//Mode Synth
//synthKeyValues = ["LFO", "Rate", "Gate", "Reverb", "Delay", "Tone", "Filter", "Cutoff", "Ressonance", "Factor"];
//LFO Amount
~yaeltex.ctls['LFO'].onChange = {|val| ("TODO: add osc function LFO amount "++val.linlin(0,127,0,1)).postln};

//Rate
~yaeltex.ctls['Rate'].onChange = {|val| ("TODO: add osc function Rate "++val.linlin(0,127,0,1)).postln};

//Mode MIR (abstract)
//BPM
~yaeltex.ctls['BPM'].onChange = {|val| ("TODO: add osc function BPM "++val.linlin(0,127,40,200).asInteger).postln};

//Key
~yaeltex.ctls['Key'].onChange = {|val| ("TODO: add osc function Key "++val.linlin(0,127,0,11).asInteger).postln};

//Master sound server PAN
//~yaeltex.ctls['Pan'].onChange = {|val| ~send_osc_all.value(~fx_pan, val.linlin(0,127,-1,1))};
~yaeltex.ctls['Pan'].onChange = {|val| ("TODO: add osc function PAN amount "++val.linlin(0,127,-1,1)).postln};

//Master sound server MASTER VOLUME
//~yaeltex.ctls['Master']onChange = {|val| ~send_osc_all.value(~fx_pan, val.linlin(0,127,-1,1))};
~yaeltex.ctls['Master'].onChange = {|val| ("TODO: add osc function Master Volume amount "++val.linlin(0,127,0,10)).postln};

/////////////////////////////////////
// 8 (knobs)
/////////////////////////////////////
//Synth
//synthKeyValues = ["LFO", "Rate",
//"Gate", "Reverb", "Delay", "Tone",
//"Filter", "Cutoff", "Ressonance", "Factor"];
//Gate
~yaeltex.ctls['Gate'].onChange ={|val| ("TODO: add osc function Gate amount "++val.linlin(0,127,0.4,1.4).asInteger).postln};

//Reverb
~yaeltex.ctls['Reverb'].onChange = {|val| ("TODO: add osc function Reverb amount "++val.linlin(0,127,0,1).asFloat).postln};

//Delay
~yaeltex.ctls['Delay'].onChange = {|val| ("TODO: add osc function Delay amount "++val.linlin(0,127,0,1).asFloat).postln};

//Tone
~yaeltex.ctls['Tone'].onChange = {|val| ("TODO: add osc function Tone amount "++val.linlin(0,127,0,1).asFloat).postln};

//last row
//Filter
~yaeltex.ctls['Filter'].onChange = {|val| ("TODO: add osc function Filter amount "++val.linlin(0,127,0,1).asFloat).postln};

//Cutoff
~yaeltex.ctls['Cutoff'].onChange = {|val| ("TODO: add osc function Cutoff amount "++val.linlin(0,127,0,1).asFloat).postln};

//Ressonance
~yaeltex.ctls['Ressonance'].onChange = {|val| ("TODO: add osc function Ressonance amount "++val.linlin(0,127,0,1).asFloat).postln};

//Factor
~yaeltex.ctls['Factor'].onChange = {|val| ("TODO: add osc function Factor amount "++val.linlin(0,127,0,1).asFloat).postln};

//MIR
//mirKeyValues = ["Key", "BPM",
//"Duration", "Pitch", "Inharmonicity", "Dissonance",
//"HFC", "PitchSalience", "SpectralCentroid", "SpectralComplexity"];
//Duration
~yaeltex.knobsMIR[2].onChange =  {|val| ("TODO: add osc function Duration amount "++val.linlin(0,127,0.1,5).asFloat).postln};

//Pitch
~yaeltex.knobsMIR[3].onChange = {|val| ("TODO: add osc function Pitch amount "++val.linlin(0,127,0,1).asFloat).postln};

//Inharmonicity
~yaeltex.knobsMIR[4].onChange = {|val| ("TODO: add osc function Inharmonicity amount "++val.linlin(0,127,0,1).asFloat).postln};

//Dissonance
~yaeltex.knobsMIR[5].onChange = {|val| ("TODO: add osc function Dissonance amount "++val.linlin(0,127,0,1).asFloat).postln};

//HFC
~yaeltex.knobsMIR[6].onChange = {|val| ("TODO: add osc function HFC amount "++val.linlin(0,127,0,1).asFloat).postln};

//PitchSalience
~yaeltex.knobsMIR[7].onChange = {|val| ("TODO: add osc function PitchSalience amount "++val.linlin(0,127,0,1).asFloat).postln};

//SpectralCentroid
~yaeltex.knobsMIR[8].onChange = {|val| ("TODO: add osc function SpectralCentroid amount "++val.linlin(0,127,0,1).asFloat).postln};

//SpectralComplexity
~yaeltex.knobsMIR[9].onChange = {|val| ("TODO: add osc function SpectralComplexity amount "++val.linlin(0,127,0,1).asFloat).postln};


////////////////
// Voices
////////////////

//TODO add a refactor an implement this logic under the Yaeltex Controller Class

// MIDI out
~default_channel = 0;
~midi_device_set_color = MIDIOut.newByName("API-cultor", "API-cultor"); //mac
//~midi_device_set_color = MIDIOut.newByName("API-cultor", "API-cultor MIDI 1"); //raspberry pi
~button_light = {
	|button_arg, onoff_arg|
	~midi_device_set_color.noteOn(~default_channel, button_arg, onoff_arg);
};

~yaeltex.voices.do {|btn, i|
	btn.onPress = {
		|val|  ("Voice"++(i+1)++" - On Press val "++val).postln;

		//all voices off
		~yaeltex.voices.do {|abtn, j|
			if( i==j, {
				~button_light.value( abtn.note, 1); //on
			},
		    {
				~button_light.value (abtn.note, 0); //off
			});
		};
	};
};

~yaeltex.voices.do {|btn, i|
	btn.onRelease = {
		|val|  ("Voice"++(i+1)++" - On Release val "++val).postln;
		~button_light.value(btn.note, 0); //off
	};
};

//Test scripts
/*
//Config
MIDIClient.destinations;
//MIDIIn.connectAll;

~yaeltex= MIDIOut.newByName("API-cultor", "API-cultor"); //usb to midi converter cable
~channel = 0;
//~eltaura.program(~channel ,msg[1]);
~yaeltex.noteOn(~channel, note: 61, veloc: 64)
~yaeltex.noteOff(~channel, note: 61, veloc: 64)

*/


// Monitor

//Mostrar MIDI input (controls)
/*
MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln;
};

//Mostrar nota + velocity
MIDIFunc.noteOn({ |veloc, num, chan, src|
	( "New note received " + num + " with vel "+veloc ).postln;
});
*/


/*
 MIDI to OSC mapping
 Yaeltex custom MIDI controller
*/
~cc_distance_sensor = 38;
~cc_bank_change = 0;

/* Program Change (PC) from voices
Example:
         ~pc_voice_1 = 81;
         ...
         ~pc_voice_8 = 88;
*/
Array.fill(8, {arg i;  i+1}).do {|i|
	('~pc_voice_'++i++'= 80+'++i).interpret;
	//('~pc_voice_'++i++' = '++('~pc_voice_'++i).interpret).postln;
};

// common/general
~cc_master = 7; //volume
~cc_pan = 10;

//MIR mode
~cc_key = 6;
~cc_bpm = 66;

~cc_duration = 100;
~cc_pitch = 101;
~cc_inharmonicity = 102;
~cc_dissonance = 103;
~cc_hfc = 104;
~cc_pitch_salience = 105;
~cc_spectral_centroid = 106;
~cc_spectral_complexity = 107;

~joy_cluster_x = 120;
~joy_cluster_y = 121;

//Synth mode
~cc_lfo_amount = 67;
~cc_rate = 76;

~cc_gate = 65;
~cc_reverb = 91;
~cc_delay = 78;
~cc_tone = 74;
~cc_type_filter = 75;
~cc_cutoff = 73;
~cc_ressonance = 71;
~cc_factor = 77;

~joy_spat_x = 122;
~joy_spat_y = 123;

//general
~joy_control_x = 124;
~joy_control_y = 125;

/*
MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln; //monitor
	switch(num,

		//fx
		~cc_master,{~send_osc.value(o, p, ~fx_volume, val.linlin(0,127,0,5))}, //linexp?
		~cc_pan,{~send_osc.value(o, p, ~fx_pan, val.linlin(0,127,-1,1))},

		//MIR descriptors (mode)
		//~cc_key = 6;
		~cc_bpm,{~send_osc.value(o, p,~mir_bpm, val.linlin(0,127,0,200))},
		~cc_duration,{~send_osc.value(o, p,~mir_duration, val.linlin(0,127,0,15))},

		//~mir_pitch
		//~cc_pitch = 101; //no es pitch centroid! WARNING
		~cc_inharmonicity,{~send_osc.value(o, p,~mir_inharmonicity, val.linlin(0,127,0,1))},
		//~cc_dissonance = 103;
		~cc_hfc,{~send_osc.value(o, p,~mir_hfc, val.linlin(0,127,0,1))},
		//~cc_pitch_salience = 105;
		~cc_spectral_centroid,{~send_osc.value(o, p,~mir_spectral_centroid, val.linlin(0,127,0,10000))},
		~cc_spectral_complexity,{~send_osc.value(o, p,~mir_spectral_complexity, val.linlin(0,127,0,1))},

		//~joy_cluster_x = 120;
		//~joy_cluster_y = 121;

		//~cc_col4_knob3,{~send_osc.value(o, p,~mir_pitch_centroid, val.linlin(0,127,0,1))},

		//Synth mode
		/*
~cc_lfo_amount = 67;
~cc_rate = 76;

~cc_gate = 65;
~cc_reverb = 91;
~cc_delay = 78;
~cc_tone = 74;
~cc_type_filter = 75;
~cc_cutoff = 73;
~cc_ressonance = 71;
~cc_factor = 77
*/
	)
};
*/

