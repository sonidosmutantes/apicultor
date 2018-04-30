//OSC Ports
~osc_port = 7000; //Open Stage Control input SYNC port
~osc_port_instrument = 9001; // cloud it instrument native port

//osc servers
~osc_server_OpenStage = NetAddr.new("127.0.0.1", ~osc_port); // Open Stage Control port
~osc_server_APICultorService = NetAddr.new("127.0.0.1", ~osc_port_instrument); //instrument port
~osc_server_SuperCollider  = NetAddr.new("127.0.0.1", NetAddr.langPort); //localhost SuperCollider port

//Downloaded file
~osc_server_OpenStage.sendMsg('/EDIT', 'soundfile', '{"value":"-", "color":"green"}');

//Voices & Leds
[1,2,3,4,5,6,7,8].do{|number|
	~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(number), '{"value":"0", "label": "", "color":"green"}');
};

//NetAddr.langPort; //osc port 57120 or 57121 TODO: add auto check to configure OpenStageControll well
//OSCFunc.trace
//OSCFunc.trace(false); // Turn posting off

//Test messages
//EDIT OpenStage UI "http://osc.ammd.net/extras/remote-control/"
//~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"BPM", "color":"green"}');
//~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"Rate", "color":"red"}');

//***************************************
//Open Stage UI
//TODO: mover esto a otro archivo separado? (es solo UI de OpenStage)
~set_mode_mir = {

	//2nd row, 2nd col
	//~osc_server_OpenStage.sendMsg('/EDIT', 'KeyLfo', '{"label":"KEY", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'KeyLfo', '{"label":"Chord Strength", "color":"green"}'); //temp?

    ~osc_server_OpenStage.sendMsg('/EDIT', 'BpmRate', '{"label":"BPM", "color":"green"}');
//FIXME: recordar valor anterior

	//3rd row
	~osc_server_OpenStage.sendMsg('/EDIT', 'DurationGate', '{"label":"DURATION", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchReverb', '{"label":"Pitch", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'InharmonicityDelay', '{"label":"Inharmonicity", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'DissonanceTone', '{"label":"Dissonance", "color":"green"}');

	//4th row
	~osc_server_OpenStage.sendMsg('/EDIT', 'HFCFilter', '{"label":"HFC", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchSalienceCutoff', '{"label":"Pitch Salience", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralCentroidRessonance', '{"label":"Spectral Centroid", "color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralComplexFactor', '{"label":"Spectral Complex", "color":"green"}');

	~osc_server_OpenStage.sendMsg('/EDIT', 'Joystick1', '{"color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Joystick2', '{"color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Up', '{"color":"green"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Down', '{"color":"green"}');

	"UI mode MIR".postln;
};

~set_mode_synth = {
	//2nd row, 2nd col
	~osc_server_OpenStage.sendMsg('/EDIT', 'KeyLfo', '{"label":"LFO Amount", "color":"orange"}');
    ~osc_server_OpenStage.sendMsg('/EDIT', 'BpmRate', '{"label":"Rate", "color":"orange"}');

//FIXME: recordar valor anterior del knob!

	//3rd row
	~osc_server_OpenStage.sendMsg('/EDIT', 'DurationGate', '{"label":"GATE", "color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchReverb', '{"label":"Reverb", "color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'InharmonicityDelay', '{"label":"Delay", "color":"orange"}');
	//~osc_server_OpenStage.sendMsg('/EDIT', 'DissonanceTone', '{"label":"Tone", "color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'DissonanceTone', '{"label":"Delay Mix", "color":"orange"}'); //temp?

	//4th row
	~osc_server_OpenStage.sendMsg('/EDIT', 'HFCFilter', '{"label":"LP/BP/HP Filter", "color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchSalienceCutoff', '{"label":"Cutoff", "color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralCentroidRessonance', '{"label":"Ressonance", "color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralComplexFactor', '{"label":"Factor", "color":"orange"}');

	~osc_server_OpenStage.sendMsg('/EDIT', 'Joystick1', '{"color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Joystick2', '{"color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Up', '{"color":"orange"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Down', '{"color":"orange"}');

	"UI mode Synth".postln;
};
//***************************************

//FIXME: recordar valor anterior del knob!
//~set_mode_mir.value;
//~set_mode_synth.value;

//sends same control value to both OSC servers (OStageC and cloud instrument)
~send_osc_all = {
  |url_arg, value_arg|

  ~osc_server_OpenStage.sendMsg(url_arg, value_arg);
  ~osc_server_APICultorService.sendMsg(url_arg, value_arg);
  ~osc_server_SuperCollider.sendMsg(url_arg, value_arg); // localhost SuperCollider
};

/////////////////////////////////////////////////////////////
//Constants
/////////////////////////////////////////////////////////////
~max_voice_volume = 4;
~delay_max = 2;
~default_delaymix = 0.5;

~max_grains_amount = 500;

/////////////////////////////////////////////////////////////
//OSC service urls
/////////////////////////////////////////////////////////////
//mode
~synth_mode = "/mode/synth";
~mir_mode = "/mode/mir";

~mir_reset = "/mir/reset";

~fx_volume = "/fx/volume"; //linexp
~fx_pan = "/fx/pan";

~fx_delay = "/fx/delay";
~fx_delaymix = "/fx/delay/mix";

~new_sound_retrieve = "/retrieve"; //TODO: change to control_x & control_y? (joystick)
~set_voices = "/set_voices"; // + integer of the voice

//~new_sound_reset = "/mir/reset"; /new
~new_sound_reset = "/clean"; //old
//~btn_control_retrieve = "/retrieve";
//~btn_control_clean = "/clean";  //-> now reset

/*
MIR paths

MIR urls schema name related with freesound (automatic translation)

Note: add new supported MIR descriptors into MIRState.py file

Check freesound mean distribution to set the range: "https://freesound.org/docs/api/analysis_docs.html#sfx-duration"
*/
~mir_duration = "/mir/duration"; //"https://freesound.org/docs/api/analysis_docs.html#sfx-duration"
~mir_pitch = "/mir/pitch/mean";
~mir_pitch_centroid = "/mir/pitch_centroid/mean"; //use pitch value or pitch centroid?
~mir_inharmonicity = "/mir/inharmonicity/mean";
~mir_dissonance = "/mir/dissonance/mean";

~mir_hfc = "/mir/hfc/mean";
~mir_pitch_salience = "/mir/pitch_salience/mean"; //"https://freesound.org/docs/api/analysis_docs.html#lowlevel-pitch-salience"

~mir_spectral_centroid = "/mir/spectral_centroid/mean";
~mir_spectral_complexity = "/mir/spectral_complexity/mean";

//~mir_key = "/mir/key"; // "https://freesound.org/docs/api/analysis_docs.html#tonal-key-key"
~mir_key = "/mir/chords_strength/mean"; //temporary

~mir_bpm = "/mir/bpm";

/////////////////////////////////
//Mode (MIR/SYNTH)
// OSC server, en esta parte porque ya existen las definiciones de los paths, etc

OSCdef.new(
	\modeMIR,
	{
		arg msg;
		//"MODE MIR".postln;
			if(msg[1].asFloat > 0,{
					~set_mode_mir.value;
					"mir mode on".postln;
						~osc_server_OpenStage.sendMsg(~mir_mode, 1);
						~osc_server_OpenStage.sendMsg(~synth_mode, 0);
				},{
					"mir mode off".postln;
					~osc_server_OpenStage.sendMsg(~mir_mode, 0);
				    ~osc_server_OpenStage.sendMsg(~synth_mode, 1);
				});


	},
	'/mode/mir',
);

OSCdef.new(
	\modeSYNTH,
	{
		arg msg;
		//"MODE SYNTH".postln;
			if(msg[1].asFloat > 0,{
					~set_mode_synth.value;
					"synth mode on".postln;
					~osc_server_OpenStage.sendMsg(~mir_mode, 0);
				    ~osc_server_OpenStage.sendMsg(~synth_mode, 1);
				},{
					"synth mode off".postln;
					~osc_server_OpenStage.sendMsg(~mir_mode, 1);
					~osc_server_OpenStage.sendMsg(~synth_mode, 0);
				});

	},
	'/mode/synth',
);



