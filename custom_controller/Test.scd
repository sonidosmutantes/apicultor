///////////////////////////////////////////////////////////////////////////
// Yaeltex custom MIDI controller
///////////////////////////////////////////////////////////////////////////

~yaeltex = YaeltexAPICultor(); //does the midiConnectAll

//Mode Synth
//LFO
~yaeltex.ctls['LFO'].onChange = {|val| ("LFO "++val.linlin(0,127,0,1)).postln};

//Rate
~yaeltex.ctls['Rate'].onChange = {|val| ("Rate "++val.linlin(0,127,0,1)).postln};

//Mode MIR (abstract)
//BPM
~yaeltex.ctls['BPM'].onChange = {|val| ("BPM "++val.linlin(0,127,40,200).asInteger).postln};

//Key
~yaeltex.ctls['Key'].onChange = {|val| ("Key "++val.linlin(0,127,0,11).asInteger).postln};

//Master sound server PAN
~yaeltex.ctls['Pan'].onChange = {|val| ("PAN "++val.linlin(0,127,-1,1)).postln};

//Master sound server MASTER VOLUME
~yaeltex.ctls['Master'].onChange = {|val| ("Master Volume "++val.linlin(0,127,0,10)).postln};

/////////////////////////////////////
// 8 (knobs)
/////////////////////////////////////
//Synth
//Gate
~yaeltex.ctls['Gate'].onChange ={|val| ("Gate "++val.linlin(0,127,0.4,1.4).asInteger).postln};

//Reverb
~yaeltex.ctls['Reverb'].onChange = {|val| ("Reverb "++val.linlin(0,127,0,1).asFloat).postln};

//Delay
~yaeltex.ctls['Delay'].onChange = {|val| ("Delay "++val.linlin(0,127,0,1).asFloat).postln};

//Tone
~yaeltex.ctls['Tone'].onChange = {|val| ("Tone "++val.linlin(0,127,0,1).asFloat).postln};

//last row
//Filter
~yaeltex.ctls['Filter'].onChange = {|val| ("Filter "++val.linlin(0,127,0,1).asFloat).postln};

//Cutoff
~yaeltex.ctls['Cutoff'].onChange = {|val| ("Cutoff "++val.linlin(0,127,0,1).asFloat).postln};

//Ressonance
~yaeltex.ctls['Ressonance'].onChange = {|val| ("Ressonance "++val.linlin(0,127,0,1).asFloat).postln};

//Factor
~yaeltex.ctls['Factor'].onChange = {|val| ("Factor "++val.linlin(0,127,0,1).asFloat).postln};

//MIR
//Duration
~yaeltex.knobsMIR[2].onChange =  {|val| ("Duration "++val.linlin(0,127,0.1,5).asFloat).postln};

//Pitch
~yaeltex.knobsMIR[3].onChange = {|val| ("Pitch "++val.linlin(0,127,0,1).asFloat).postln};

//Inharmonicity
~yaeltex.knobsMIR[4].onChange = {|val| ("Inharmonicity "++val.linlin(0,127,0,1).asFloat).postln};

//Dissonance
~yaeltex.knobsMIR[5].onChange = {|val| ("Dissonance "++val.linlin(0,127,0,1).asFloat).postln};

//HFC
~yaeltex.knobsMIR[6].onChange = {|val| ("HFC "++val.linlin(0,127,0,1).asFloat).postln};

//PitchSalience
~yaeltex.knobsMIR[7].onChange = {|val| ("PitchSalience "++val.linlin(0,127,0,1).asFloat).postln};

//SpectralCentroid
~yaeltex.knobsMIR[8].onChange = {|val| ("SpectralCentroid "++val.linlin(0,127,0,1).asFloat).postln};

//SpectralComplexity
~yaeltex.knobsMIR[9].onChange = {|val| ("SpectralComplexity "++val.linlin(0,127,0,1).asFloat).postln};


////////////////
// Voices buttons
////////////////

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


