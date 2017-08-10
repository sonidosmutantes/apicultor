//MIDIIn.connectAll;

//OSC
~osc_port = 7000; //open stage input sync port
~osc_port_instrument = 9001; // clout it instrument native port
o = NetAddr.new("127.0.0.1", ~osc_port);
p = NetAddr.new("127.0.0.1", ~osc_port_instrument);

~send_osc = {
  |a_arg, b_arg, url_arg, value_arg|
  a_arg.sendMsg(url_arg, value_arg);
  b_arg.sendMsg(url_arg, value_arg);
};

~send_osc_push_button = {
	//push button (not toggle) sends 1 value waits 0.3 seconds and send 0 value
  |a_arg, b_arg, url_arg|

//  a_arg.sendMsg(url_arg, 1); //on
//  b_arg.sendMsg(url_arg, 1); //on
  //0.3.wait;
 a_arg.sendMsg(url_arg, 0); //off
 b_arg.sendMsg(url_arg, 0); //off
};

// Test messages
//~send_osc_push_button.value(o, p,~btn_control_retrieve);
//~send_osc.value(o, p,"/fx/volume", 1.1); //test msg
//~send_osc.value(o, p, "/fx/volume", 127, 10); //test msg

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

~fx_volume = "/fx/volume"; //linexp
~fx_pan = "/fx/pan";

~mir_duration = "/mir/duration";
~mir_bpm = "/mir/bpm";

~mir_inharmonicity = "/mir/inharmonicity/mean";
~mir_hfc = "/mir/hfc/mean";
~mir_spectral_complexity = "/mir/spectral_complexity/mean";
~mir_spectral_centroid = "/mir/spectral_centroid/mean";
~mir_pitch_centroid = "/mir/pitch_centroid/mean";

MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln; //monitor
	switch(num,
		//fx
		~cc_slider_master,{~send_osc.value(o, p, ~fx_volume, val.linlin(0,127,0,5))}, //linexp?
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

~mir_spectral_centroid_enabled = "/mir/spectral_centroid/mean/enabled";
~mir_duration_enabled = "/mir/duration/enabled";
~mir_bpm_enabled = "/mir/bpm/enabled";

~mir_inharmonicity_enabled = "/mir/inharmonicity/mean/enabled";
~mir_hfc_enabled = "/mir/hfc/mean/enabled";
~mir_spectral_complexity_enabled = "/mir/spectral_complexity/mean/enabled";
~mir_pitch_centroid_enabled = "/mir/pitch_centroid/mean/enabled";

//button
~btn_control_retrieve = "/retrieve";
~btn_control_clean = "/clean";

MIDIFunc.noteOn({ |veloc, num, chan, src|
	[chan,num,veloc].postln; //monitor
	switch(num,
		~noteon_solo_master,{
			//~send_osc_push_button.value(o, p,~btn_control_retrieve);
			~send_osc.value(o, p, ~btn_control_retrieve, 1); //sends 1
		},

		//sends descriptor turn off (not enabled, 0)
		~noteon_solo_1,{
			~send_osc_push_button.value(o, p,~mir_hfc_enabled);
		},
		~noteon_solo_2,{
			~send_osc_push_button.value(o, p,~mir_spectral_complexity_enabled);
		},
		~noteon_solo_3,{
			~send_osc_push_button.value(o, p,~mir_inharmonicity_enabled);
		},
	    ~noteon_solo_4,{
			~send_osc_push_button.value(o, p,~mir_pitch_centroid_enabled);
		},

	    ~noteon_recarm_1,{
			~send_osc_push_button.value(o, p,~mir_spectral_centroid_enabled);
		},
	    ~noteon_recarm_2,{
			~send_osc_push_button.value(o, p,~mir_duration_enabled);
		},
	    ~noteon_recarm_3,{
			~send_osc_push_button.value(o, p,~mir_bpm_enabled);
		},
	)
});

//MIDIIn.connectAll;

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

