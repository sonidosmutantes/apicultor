MIDIIn.connectAll;
0.5.wait;
"midi_to_osc.scd".loadRelative;

///////////////////////////////////////////////////////////////////////////
// Yaeltex custom MIDI controller
///////////////////////////////////////////////////////////////////////////
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

/*
NetAddr.langPort; //osc port 57120 ó 57121

//ver todos los mensajes
OSCFunc.trace
OSCFunc.trace(false); // Turn posting off
*/

