//OSC Ports
~osc_port = 7000; //Open Stage Control input SYNC port
~osc_port_instrument = 9001; // cloud it instrument native port

~network_broadcast = "192.168.1.255";
NetAddr.broadcastFlag = true; //Zero conf version. broadcast to the network
//osc servers
//Broadcast IP to avoid setting fixed IP (of the OpenStageControl server)
~osc_server_OpenStage = NetAddr.new(~network_broadcast, ~osc_port); //Broadcast to Open Stage
//~osc_server_OpenStage = NetAddr.new("127.0.0.1", ~osc_port); // Open Stage Control port

~osc_server_APICultorService = NetAddr.new("127.0.0.1", ~osc_port_instrument); //instrument port
~osc_server_SuperCollider  = NetAddr.new("127.0.0.1", NetAddr.langPort); //localhost SuperCollider port

//OSCFunc.trace
//OSCFunc.trace(false); // Turn posting off

("Listening OSC Port: "++NetAddr.langPort).postln;

/////////////////////////////////////////////////////////////
//Constants
/////////////////////////////////////////////////////////////
~max_voice_volume = 6;
~delay_max = 2;
~default_delaymix = 0.5;

~max_grains_amount = 500;
~master_volume_max = 25;

//***************************************
//Open Stage UI

//Test messages
//EDIT OpenStage UI "http://osc.ammd.net/extras/remote-control/"
//~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"BPM", "color":"green"}');
//~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"Rate", "color":"red"}');

//TODO: mover esto a otro archivo separado? (es solo UI de OpenStage)

//Downloaded file
~osc_server_OpenStage.sendMsg('/EDIT', 'soundfile', '{"value":"-", "color":"green"}');

//Voices & Leds
[1,2,3,4,5,6,7,8].do{|number|
	~osc_server_OpenStage.sendMsg('/EDIT', 'led_'++(number), '{"value":"0", "label": "", "color":"green"}');
};

//Test modes
//~set_mode_mir.value();
//~set_mode_synth.value();

/*
Disable to work with a unique UI in stage control
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
	~osc_server_OpenStage.sendMsg('/EDIT', 'KeyLfo', '{"label":"LFO Amount", "color":"blue"}');
    ~osc_server_OpenStage.sendMsg('/EDIT', 'BpmRate', '{"label":"Rate", "color":"blue"}');

//FIXME: recordar valor anterior del knob!

	//3rd row
	~osc_server_OpenStage.sendMsg('/EDIT', 'DurationGate', '{"label":"GATE", "color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchReverb', '{"label":"Reverb", "color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'InharmonicityDelay', '{"label":"Delay", "color":"blue"}');
	//~osc_server_OpenStage.sendMsg('/EDIT', 'DissonanceTone', '{"label":"Tone", "color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'DissonanceTone', '{"label":"Delay Mix", "color":"blue"}'); //temp?

	//4th row
	~osc_server_OpenStage.sendMsg('/EDIT', 'HFCFilter', '{"label":"LP/BP/HP Filter", "color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'PitchSalienceCutoff', '{"label":"Cutoff", "color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralCentroidRessonance', '{"label":"Ressonance", "color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'SpectralComplexFactor', '{"label":"Factor", "color":"blue"}');

	~osc_server_OpenStage.sendMsg('/EDIT', 'Joystick1', '{"color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Joystick2', '{"color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Up', '{"color":"blue"}');
	~osc_server_OpenStage.sendMsg('/EDIT', 'Down', '{"color":"blue"}');

	"UI mode Synth".postln;
};

//FIXME: recordar valor anterior del knob!
*/

// General send
//sends same control value to both OSC servers (OStageC and cloud instrument)
/*
~send_osc_all = {
  |url_arg, value_arg|

  ~osc_server_OpenStage.sendMsg(url_arg, value_arg);
  ~osc_server_APICultorService.sendMsg(url_arg, value_arg);
  ~osc_server_SuperCollider.sendMsg(url_arg, value_arg); // localhost SuperCollider
};
*/

/////////////////////////////////////////////////////////////
//OSC service urls
/////////////////////////////////////////////////////////////
//mode
~synth_mode = "/mode/synth";
~mir_mode = "/mode/mir";

//FX & Synth
~fx_volume = "/fx/volume"; //linexp
~fx_pan = "/fx/pan";

~fx_delay = "/fx/delay";
~fx_delaymix = "/fx/delay/mix";

~fx_lfo = "/fx/lfo";
~fx_rate = "/fx/rate";

~fx_gate = "/fx/gate";
~fx_reverb = "/fx/reverb";
~fx_filter = "/fx/filter";
~fx_cutoff = "/fx/cutoff";
~fx_ressonance = "/fx/ressonance";
~fx_factor = "/fx/factor";

~new_sound_retrieve = "/retrieve"; //TODO: change to control_x & control_y? (joystick)
~mir_reset = "/mir/reset"; // Resets MIR state in the service
//~new_sound_reset = "/clean"; //old


~set_voices = "/set_voices"; //Set active voice (fx & mir) with integer of the voice as argument

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


/*
//Monitor MIDI input (controls)
MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln;
};

//Monitor note + velocity
MIDIFunc.noteOn({ |veloc, num, chan, src|
	( "New note received " + num + " with vel "+veloc ).postln;
});
*/
