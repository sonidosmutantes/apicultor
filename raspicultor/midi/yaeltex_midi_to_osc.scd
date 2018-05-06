//MIDIIn.connectAll; //"MIDIEndPoint("API-cultor", "API-cultor")"
//0.5.wait;
"midi_to_osc.scd".loadRelative;
~yaeltex = YaeltexAPICultor(); //también hace el midi connect All

// MIDI out
~default_channel = 0;
if( ~host=="mac", {
	~midi_device_set_color = MIDIOut.newByName("API-cultor", "API-cultor"); //mac
}
,{
	~midi_device_set_color = MIDIOut.newByName("API-cultor", "API-cultor MIDI 1"); //raspberry pi
});


///////////////////////////////////////////////////////////////////////////
// Yaeltex custom MIDI controller
///////////////////////////////////////////////////////////////////////////
//Note: FX's sends message to osc_server_SuperCollider and MIR sends to osc_server_APICultorService

// First row of 4 knobs (into the right)
//Mode Synth
//synthKeyValues = ["LFO", "Rate", "Gate", "Reverb", "Delay", "Tone", "Filter", "Cutoff", "Ressonance", "Factor"];
//LFO (rate, not amount)
~yaeltex.ctls['LFO'].onChange = {|val|
	val = val.linlin(0,127,0,100).asFloat;
	//("SYNTH: LFO "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_lfo, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'LFO', '{"value":"'++val++'"}');
};

//Rate  (Pitch is 0..4 range)
~yaeltex.ctls['Rate'].onChange = {|val|
	val = val.linlin(0,127,0,4).asFloat;
	//("SYNTH: Rate "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_rate, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Rate', '{"value":"'++val++'"}');
};

//Mode MIR (abstract)
//BPM
~yaeltex.ctls['BPM'].onChange = {|val|
	val = val.linlin(0,127,40,200).asFloat;
	("MIR: BPM "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_bpm, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"value":"'++val++'"}');
};

//Key
~yaeltex.ctls['Key'].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Key (now chord strength) "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_key, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Key', '{"value":"'++val++'"}');
};

//Master sound server PAN
~yaeltex.ctls['Pan'].onChange = {|val|
	val = val.linlin(0,127,-1,1).asFloat;
	~osc_server_SuperCollider.sendMsg(~fx_pan, val);
	~osc_server_OpenStage.sendMsg('/EDIT', 'Pan', '{"value":"'++val++'"}');
};

//Master sound server MASTER VOLUME
~yaeltex.ctls['Master'].onChange = {|val|
	val = val.linlin(0,127,0,~master_volume_max).asFloat; //7 are -20dB
	~osc_server_SuperCollider.sendMsg(~fx_volume, val);

	//UI -> range: 0..10
	//~osc_server_OpenStage.sendMsg('/EDIT', 'Master', '{"value":"'++val.linlin(0,~master_volume_max,0,10)++'"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Master', '{"value":"'++val.linlin(0,~master_volume_max,0,10)++'"}');
};

/////////////////////////////////////
// 8 (knobs)
/////////////////////////////////////
//Synth
//synthKeyValues = ["LFO", "Rate",
//"Gate", "Reverb", "Delay", "Tone",
//"Filter", "Cutoff", "Ressonance", "Factor"];

//Gate
~yaeltex.ctls['Gate'].onChange ={|val|
	val = val.linlin(0,127,0,1).asFloat;
	//("SYNTH: Gate "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_gate, val);

	if( val>0, {val=1}, {val=0});

	~osc_server_OpenStage.sendMsg('/EDIT', 'Gate', '{"value":"'++val++'"}');
};

//Reverb
~yaeltex.ctls['Reverb'].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;

	//("SYNTH: Reverb "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_reverb, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Reverb', '{"value":"'++val++'"}');
};

//Delay
~yaeltex.ctls['Delay'].onChange = {|val|
	val = val.linlin(0,127,0,~delay_max).asFloat;
	//("SYNTH: Delay "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_delay, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Delay', '{"value":"'++val++'"}');
};

//Tone (temporary delaymix)
~yaeltex.ctls['Tone'].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	//("SYNTH: Tone (delaymix by now) "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_delaymix, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Tone', '{"value":"'++val++'"}');
};

//last row
//Filter (LP/BP/HP) -> 0-> OFF // 1:LPF,2:BPF,3:HPF
~yaeltex.ctls['Filter'].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	if( val<0.1, { val=0 }, { if( val<0.38, {val=1},{ if( val<0.68, {val=2}, {val=3})})});
	//("SYNTH: Filter Mode "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_filter, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Filter', '{"value":"'++val++'"}');
};

//Cutoff
~yaeltex.ctls['Cutoff'].onChange = {|val|
	val = val.linlin(0,127,20,5000).asFloat;
	//("SYNTH: Cutoff "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_cutoff, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Cutoff', '{"value":"'++val++'"}');
};

//Ressonance
~yaeltex.ctls['Ressonance'].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	//("SYNTH: Ressonance "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_ressonance, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Ressonance', '{"value":"'++val++'"}');
};

//Factor
~yaeltex.ctls['Factor'].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	//("SYNTH: Factor "++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_factor, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Factor', '{"value":"'++val++'"}');
};

//MIR
//mirKeyValues = ["Key", "BPM",
//"Duration", "Pitch", "Inharmonicity", "Dissonance",
//"HFC", "PitchSalience", "SpectralCentroid", "SpectralComplexity"];
//Duration
~yaeltex.knobsMIR[2].onChange =  {|val|
	val = val.linlin(0,127,0.1,5).asFloat; //TODO: define range
	("MIR: Duration "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_duration, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Duration', '{"value":"'++val++'"}');
};

//Pitch
~yaeltex.knobsMIR[3].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Pitch "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_pitch, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Pitch', '{"value":"'++val++'"}');
};

//Inharmonicity
~yaeltex.knobsMIR[4].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Inharmonicity "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_inharmonicity, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Inharmonicity', '{"value":"'++val++'"}');
};

//Dissonance
~yaeltex.knobsMIR[5].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: Dissonance "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_dissonance, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Dissonance', '{"value":"'++val++'"}');
};

//HFC
~yaeltex.knobsMIR[6].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: HFC "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_hfc, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'HFC', '{"value":"'++val++'"}');
};

//PitchSalience
~yaeltex.knobsMIR[7].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: PitchSalience "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_pitch_salience, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchSalience', '{"value":"'++val++'"}');
};

//SpectralCentroid
~yaeltex.knobsMIR[8].onChange = {|val|
	val = val.linlin(0,127,0,10000).asFloat;
	("MIR: SpectralCentroid "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_spectral_centroid, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralCentroid', '{"value":"'++val++'"}');
};

//SpectralComplexity
~yaeltex.knobsMIR[9].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	("MIR: SpectralComplexity "++val).postln;
	~osc_server_APICultorService.sendMsg(~mir_spectral_complexity, val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralComplexity', '{"value":"'++val++'"}');
};

////////////////
// Voices
////////////////

//midi out
~button_light = {
	|button_arg, onoff_arg|
	~midi_device_set_color.noteOn(~default_channel, button_arg, onoff_arg);
};

~button_blink = {
   |button_arg|
	a = Task({ //3 times blink
		3.do { arg i;
			~button_light.value(button_arg, 1);
			0.3.wait;
			~button_light.value(button_arg, 0);
			0.3.wait;
		}
	});
	a.start;
};

~yaeltex.voices.do {|btn, i|
	btn.onPress = {
		|val|  ("Voice"++(i+1)++" - On Press val "++val).postln;
		~osc_server_SuperCollider.sendMsg(~set_voices, i+1);

		~yaeltex.voices.do {|abtn, j|
			if( i==j, {
				~button_light.value( abtn.note, 1); //on
			},
		    {	//all voices off
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

//////////////////////////////////////
//Distance sensor -> Grains amount
//////////////////////////////////////
~yaeltex.distanceSensor[0].onChange = {|val|
	val = val.linlin(0,127,0,~max_grains_amount).asFloat;
	~osc_server_SuperCollider.sendMsg("/synth/granular/grains/amount", val);

	~osc_server_OpenStage.sendMsg('/EDIT', 'Grains', '{"value":"'++val++'"}');
};

//////////////////////////////////////
//Joysticks config
//////////////////////////////////////

//Synth MODE
//Joystick 1 (left)  without return
//X: PAN
~yaeltex.ctls['joy1Synth_x'].onChange = {|val|
//~yaeltex.joyNoReturnSynth[0].onChange = {|val|
	//("Joystick1 (no-return) x value:"++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_pan, val.linlin(0,127,-1,1))
};
//Y: VOLUME
~yaeltex.ctls['joy1Synth_y'].onChange = {|val|
//~yaeltex.joyNoReturnSynth[1].onChange = {|val|
	val = val.linlin(0,127,1,0).asFloat;
	//("Joystick1 (no-return) y value:"++val).postln;
	~osc_server_SuperCollider.sendMsg("/voice_volume", val.linlin(0,1,0,~max_voice_volume) );
};

//Joyttick 2 (right) with-return
//X: Trigger grains
//~yaeltex.ctls['joy2Synth_x'].onChange = {|val|
~yaeltex.joyReturnSynth[0].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
//	("Joystick2 (with-return) x value:"++val).postln;
	if( val==1, { //to the rigth
		~osc_server_SuperCollider.sendMsg("/synth/granular/trigger", 1);
	});
};
//Y: Freeze ON/OFF
//~yaeltex.ctls['joy2Synth_y'].onChange = {|val|
~yaeltex.joyReturnSynth[1].onChange = {|val|
	val = val.linlin(0,127,1,0).asFloat;
	//("Joystick2 (with-return) y value:"++val).postln;
	if( val==1, {
		~osc_server_SuperCollider.sendMsg("/synth/freeze", 1, -1 ); //ON , -1 means active_voice
	});
	if( val==0, {
		~osc_server_SuperCollider.sendMsg("/synth/freeze", 0, -1 ); //OFF , -1 means active_voice
	});
};


//MIR MODE
//Joystick 1 (left)  without return
//X: PAN
~yaeltex.joyNoReturnMIR[0].onChange = {|val|
	//val = val.linlin(0,127,0,1).asFloat;
	//("Joystick1 (no-return) x value:"++val).postln;
	~osc_server_SuperCollider.sendMsg(~fx_pan, val.linlin(0,127,-1,1))
};
//Y: Volume
~yaeltex.joyNoReturnMIR[1].onChange = {|val|
	val = val.linlin(0,127,1,0).asFloat;
	//("Joystick1 (no-return) y value:"++val).postln;
	~osc_server_SuperCollider.sendMsg("/voice_volume", val.linlin(0,1,0,~max_voice_volume) );
};

//Joyttick 2 (right) with-return
//X: Nothing
~yaeltex.joyReturnMIR[0].onChange = {|val|
	val = val.linlin(0,127,0,1).asFloat;
	//("Joystick2 (with-return) x value:"++val).postln;
	//Not immplemented
};
//Y: Retrieve new sound / Reset descriptors
~yaeltex.joyReturnMIR[1].onChange = {|val|
	val = val.linlin(0,127,1,0).asFloat;
	//("Joystick2 (with-return) y value:"++val).postln;
	//Freeze ON/OFF
	if( val==1, {
		~osc_server_SuperCollider.sendMsg(~new_sound_retrieve, 1); //sends 1
	});
	if( val==0, {
		~osc_server_SuperCollider.sendMsg(~mir_reset, 1);
	});
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

/*
MIDI Documentation

//Common/general
~cc_master = 7; //volume
~cc_pan = 10;

~cc_distance_sensor = 38;

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
*/
