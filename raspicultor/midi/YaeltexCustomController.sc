// =======================================================
// Title         :  Yaeltex Custom Controller API-Cultor
// Description   : Controller class for API-Cultor Yaeltex MIDI controller
// Version       :   1.0
// Copyright (c) : Hernán Ordiales (h@ordia.com.ar) 2018, basado en código de "https://github.com/davidgranstrom/ArturiaBeatStep"
// =======================================================

//TODO: add a demo file (.scd)
// + planilla con la spec (CC + Notes)

YaeltexAPICultor {
	var <master, <pan, <distanceSensor, <knobsMIR, <joyNoReturnMIR, <joyReturnMIR, <knobsSynth, <joyNoReturnSynth, <joyReturnSynth, <voices;

    var <ctls;
	var mirKeyValues, synthKeyValues, joyKeyValues;
    var masterValue, panValue, distanceSensorValue, knobsMIRValues, joyNoReturnMIRValues, joyReturnMIRValues, knobsSynthValues, joyNoReturnSynthValues, joyReturnSynthValues, voicesValues;

    *new {
        ^super.new.init;
    }

    init {
        ctls = ();

		//vars
		master = List[];
		pan = List[];

		distanceSensor = List[];

		knobsMIR = List[];
		joyNoReturnMIR = List[];
        joyReturnMIR = List[];

		knobsSynth = List[];
		joyNoReturnSynth= List[];
        joyReturnSynth = List[];

		voices = List[];

		//MIDI values (CC & Notes)
		masterValue = [7];
		panValue = [10];

		distanceSensorValue = [38];

		//BPM, Key, Duration, Pitch, Inharmonicity, Dissonance, HFC, Pitch Salience, Spectral Centroid, Spectral Complexity
		knobsMIRValues    = [ 6, 66, 102, 103, 104, 105, 106, 107, 108, 109  ];

		//LFO Amount, RATE, Gate, Reverb, Delay, Tone, LP/BP/HP Filter, Cutoff, Ressonance, Factor
	    knobsSynthValues = [ 67, 76, 65  , 91  , 78   , 74,    75,   73,   71,  77];

		voicesValues = [ 60,61,62,63,64,65,66,67 ]; //Voice1 to Voice8

		joyNoReturnMIRValues    = [110, 111]; //x, y
		joyReturnMIRValues    = [112, 113]; //x, y

		joyNoReturnSynthValues = [114, 115]; //x, y
		joyReturnSynthValues = [116, 117]; //x, y

        MIDIClient.init;
        MIDIIn.connectAll;

        this.assignCtls;

    }

    assignCtls {
		mirKeyValues = ["BPM", "Key", "Duration", "Pitch", "Inharmonicity", "Dissonance", "HFC", "PitchSalience", "SpectralCentroid", "SpectralComplexity"];

        knobsMIRValues.do {|cc, i|
            //var key  = ("knobMIR" ++ (i+1)).asSymbol;
			var key  = mirKeyValues[i].asSymbol;
            var knob = YaeltexKnob(key, cc);
            knobsMIR.add(knob);
            ctls.put(key, knob);
        };

		synthKeyValues = ["LFO", "Rate", "Gate", "Reverb", "Delay", "Tone", "Filter", "Cutoff", "Ressonance", "Factor"];

		knobsSynthValues.do {|cc, i|
            //var key  = ("knobSynth" ++ (i+1)).asSymbol;
			var key  = synthKeyValues[i].asSymbol;
            var knob = YaeltexKnob(key, cc);
            knobsSynth.add(knob);
            ctls.put(key, knob);
        };

		masterValue.do {|note, i|
            var key = ("Master").asSymbol;
            var knob = YaeltexKnob(key, note);
            master.add(knob);
            ctls.put(key, knob);
        };

		panValue.do {|note, i|
            var key = ("Pan").asSymbol;
            var knob = YaeltexKnob(key, note);
            pan.add(knob);
            ctls.put(key, knob);
        };

		distanceSensorValue.do {|note, i|
            var key = ("DistanceSensor").asSymbol;
            var knob = YaeltexKnob(key, note);
            distanceSensor.add(knob);
            ctls.put(key, knob);
        };

		voicesValues.collect {|note, i|
            var key = ("Voice" ++ (i+1)).asSymbol;
            var btn = YaeltexToggleButton(key, note);
            voices.add(btn);
            ctls.put(key, btn);
        };

		joyKeyValues = ["joy1Synth_x", "joy1Synth_y", "joy2Synth_x", "joy2Synth_y", "joy1Mir_x", "joy1Mir_y", "joy2Mir_x", "joy2Mir_y"];

		joyNoReturnSynthValues.collect {|note, i|
			var key = joyKeyValues[0..1][i].asSymbol;
            var joystick = YaeltexKnob(key, note);
            joyNoReturnSynth.add(joystick);
            ctls.put(key, joystick);
		};
	    joyNoReturnMIRValues.collect {|note, i|
			var key = joyKeyValues[2..3][i].asSymbol;
            var joystick = YaeltexKnob(key, note);
            joyNoReturnMIR.add(joystick);
            ctls.put(key, joystick);
		};
		joyReturnSynthValues.collect {|note, i|
			var key = joyKeyValues[4..5][i].asSymbol;
            var joystick = YaeltexKnob(key, note);
            joyReturnSynth.add(joystick);
            ctls.put(key, joystick);
		};
	    joyReturnMIRValues.collect {|note, i|
			var key = joyKeyValues[6..7][i].asSymbol;
            var joystick = YaeltexKnob(key, note);
            joyReturnMIR.add(joystick);
            ctls.put(key, joystick);
		};
    }

    freeAll {
        ctls.do(_.free);
    }

    doesNotUnderstand {|selector ... args|
        ^ctls[selector] ?? { ^super.doesNotUnderstand(selector, args) }
    }
}

YaeltexKnob {
    var key, cc;

    *new {|key, cc|
        ^super.newCopyArgs(("abs_" ++ key).asSymbol, cc);
    }

    onChange_ {|func|
        MIDIdef.cc(key, func, cc);
    }

    free {
        MIDIdef.cc(key).free;
    }
}

YaeltexToggleButton {
    var key, <note; //note has a getter

    *new {|key, note|
        ^super.newCopyArgs("abs_" ++ key, note);
    }

    onPress_ {|func|
        MIDIdef.noteOn((key ++ "_on").asSymbol, {|val| func.(val) }, note);
    }

    onRelease_ {|func|
        MIDIdef.noteOff((key ++ "_off").asSymbol, {|val| func.(val) }, note);
    }

    onChange_ {|func|
        MIDIdef.noteOn((key ++ "_on_change").asSymbol, {|val| func.(val) }, note);
        MIDIdef.noteOff((key ++ "_off_change").asSymbol, {|val| func.(val) }, note);
    }

    free {
        var labels = [ "_on", "_off", "_on_change", "_off_change" ];

        labels.do {|label|
            var k = (key ++ label).asSymbol;
            MIDIdef.cc(k).free;
        };
    }
}
