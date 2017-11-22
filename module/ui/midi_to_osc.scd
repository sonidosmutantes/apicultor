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

~cc_col8_knob3 = 60;
~cc_slider5= 49;
~cc_slider6= 53;
~cc_slider7= 57;
~cc_slider8= 61;

~cc_slider_master = 62;

~fx_volume = "/fx/volume"; //linexp
~fx_pan = "/fx/pan";

~mir_duration = "/mir/duration";
~mir_bpm = "/mir/bpm";
~mir_inharmonicity = "/mir/inharmonicity";

~mir_hfc = "/mir/hfc/mean";
~mir_spectral_complexity = "/mir/spectral_complexity/mean";
~mir_spectral_centroid = "/mir/spectral_centroid/mean";
~mir_pitch_centroid = "/mir/pitch_centroid/mean";

MIDIIn.control = {arg src, chan, num, val;
	[chan,num,val].postln; //monitor
	switch(num,
		//fx
		~cc_slider_master,{~send_osc.value(o, p, ~fx_volume, val.linlin(0,127,0,10))}, //linexp?
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
~noteon_solo = 27;

// solo buttons
~noteon_solo_1 = 1;
~noteon_solo_2 = 4;
~noteon_solo_3 = 7;
~noteon_solo_4 = 10;
~noteon_solo_5 = 13;
~noteon_solo_6 = 16;
~noteon_solo_7 = 19;
~noteon_solo_8 = 22;

~mir_spectral_centroid_enabled = "/mir/spectral_centroid/mean/enabled";
~mir_duration_enabled = "/mir/duration/enabled";
~mir_bpm_enabled = "/mir/bpm/enabled";

~btn_control_retrieve = "/retrieve";
~btn_control_clean = "/clean";
MIDIFunc.noteOn({ |veloc, num, chan, src|
	[chan,num,veloc].postln; //monitor
	switch(num,
		~noteon_solo,{
			~send_osc.value(o, p,~btn_control_retrieve, 1);0.3.wait;~send_osc.value(o, p,~osc_control_retrieve, 0)
		},
		~noteon_solo_1,{
			~send_osc.value(o, p,~mir_spectral_centroid_enabled, 1);0.3.wait;~send_osc.value(o, p,~mir_spectral_centroid_enabled, 0)
		},
		~noteon_solo_2,{
			~send_osc.value(o, p,~mir_duration_enabled, 1);0.3.wait;~send_osc.value(o, p,~mir_duration_enabled, 0)
		},
		~noteon_solo_3,{
			~send_osc.value(o, p,~mir_bpm_enabled, 1);0.3.wait;~send_osc.value(o, p,~mir_bpm_enabled, 0)
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

