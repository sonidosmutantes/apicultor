
//OSC
~osc_port = 7000; //Open Stage Control input SYNC port
~osc_port_instrument = 9001; // cloud it instrument native port
~osc_server_OpenStage = NetAddr.new("127.0.0.1", ~osc_port); // Open Stage Control port
~osc_server_APICultorService = NetAddr.new("127.0.0.1", ~osc_port_instrument); //instrument port
~osc_server_SuperCollider  = NetAddr.new("127.0.0.1", NetAddr.langPort); //localhost SuperCollider port


//NetAddr.langPort; //osc port 57120 or 57121 TODO: add auto check to configure OpenStageControll well
//OSCFunc.trace
//OSCFunc.trace(false); // Turn posting off

//Test messages
//EDIT OpenStage UI "http://osc.ammd.net/extras/remote-control/"
//~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"BPM", "color":"green"}');
//~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"Rate", "color":"red"}');

~set_mode_mir = {

	//2nd row, 2nd col
	~osc_server_OpenStage.sendMsg('/EDIT', 'KEY', '{"label":"KEY", "color":"green"}');
    ~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"BPM", "color":"green"}');
//FIXME: recordar valor anterior

	//3rd row
	~osc_server_OpenStage.sendMsg('/EDIT', 'Duration', '{"label":"DURATION", "color":"green"}');
	"UI mode MIR".postln;
};

~set_mode_synth = {
	//2nd row, 2nd col
	~osc_server_OpenStage.sendMsg('/EDIT', 'KEY', '{"label":"LFO Amount", "color":"red"}');
    ~osc_server_OpenStage.sendMsg('/EDIT', 'BPM', '{"label":"Rate", "color":"red"}');

//FIXME: recordar valor anterior del knob!

	//3rd row
	~osc_server_OpenStage.sendMsg('/EDIT', 'Duration', '{"label":"GATE", "color":"red"}');

	"UI mode Synth".postln;
};

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

//OSC service urls

//mode
~synth_mode = "/mode/synth";
~mir_mode = "/mode/mir";

~fx_volume = "/fx/volume"; //linexp
~fx_pan = "/fx/pan";

~new_sound_retrieve = "/retrieve"; //TODO: change to control_x & control_y? (joystick)
//~new_sound_reset = "/mir/reset"; /new
~new_sound_reset = "/clean"; //old
//~btn_control_retrieve = "/retrieve";
//~btn_control_clean = "/clean";  //-> now reset

// MIR urls schema name related with freesound (automatic translation)
~mir_duration = "/mir/duration";
//pitch
//~mir_pitch_centroid = "/mir/pitch_centroid/mean";
//TODO: use pitch value, not pitch centroid?

~mir_inharmonicity = "/mir/inharmonicity/mean";
//TODO: dissonance

//TODO: key
~mir_bpm = "/mir/bpm";

~mir_hfc = "/mir/hfc/mean";
//TODO test/develop: pitchsalience
~mir_pitchsalience = "/mir/pitchsalience"; //TODO: implement in server
~mir_spectral_centroid = "/mir/spectral_centroid/mean";
~mir_spectral_complexity = "/mir/spectral_complexity/mean";



