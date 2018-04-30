// =======================================================
// Title         :  iRig PADS, MPK mini, AkaiMidiMix, etc based on ArturiaBeatStep
// Description   : Controller class for iRig PADS, AkaiMidiMix
// Version       :  Modificado por hordia 2017, 2018
// Copyright (c) : basado en código de "https://github.com/davidgranstrom/ArturiaBeatStep"
// =======================================================

//TODO: connect only the selected device
// input MIDI connect by name -> ver "http://new-supercollider-mailing-lists-forums-use-these.2681727.n2.nabble.com/MIDIIn-connect-by-name-td7583043.html"

AkaiMidiMix {
	var <master, <knobsA, <knobsB, <knobsC, <sliders, <recpads, <mutepads, <bankLeft, <bankRight, <solo;

    var ctls;
    var knobAValues, knobBValues, knobCValues, sliderValues, muteValues, recValues, masterValue, bankLeftValue, bankRightValue, soloValue;

    *new {
        ^super.new.init;
    }

    init {
        ctls = ();

		bankLeft = List[];
		bankRight = List[];
        master = List[];
		solo = List[];

		knobsA = List[];
		knobsB = List[];
		knobsC = List[];
		sliders = List[];

		mutepads  = List[];
        recpads  = List[];


        //Knobs: Row 1,2,3
		//Slides: Row 4
	    knobAValues = [ 16,20,24,28,46,50,54,58 ];
		knobBValues = [ 17,21,25,29,47,51,55,59 ];
		knobCValues = [ 18,22,26,30,48,52,56,60 ];
		sliderValues  = [ 19,23,27,31,49,53,57,61 ];

		/*
		// Row1
		// Row2
		// Col1
        padValues  = [
			1,4,7,10,13,16,19,22,
			3,6,9,12,15,18,21,24,
			25,26,27
		];*/

		muteValues  = [
			1,4,7,10,13,16,19,22
		];
		recValues  = [
			3,6,9,12,15,18,21,24
		];

		bankLeftValue = [25];
		bankRightValue = [26];
		soloValue = [27];
		masterValue = [62]; 	//MASTER slide

        MIDIClient.init;
        MIDIIn.connectAll;

        this.assignCtls;
    }

    assignCtls {
        knobAValues.do {|cc, i|
            var key  = ("knobA" ++ (i+1)).asSymbol;
            var knob = ABSKnob(key, cc);
            knobsA.add(knob);
            ctls.put(key, knob);
        };
        knobBValues.do {|cc, i|
            var key  = ("knobB" ++ (i+1)).asSymbol;
            var knob = ABSKnob(key, cc);
            knobsB.add(knob);
            ctls.put(key, knob);
        };
		knobCValues.do {|cc, i|
            var key  = ("knobC" ++ (i+1)).asSymbol;
            var knob = ABSKnob(key, cc);
            knobsC.add(knob);
            ctls.put(key, knob);
        };
		sliderValues.do {|cc, i|
            var key  = ("slider" ++ (i+1)).asSymbol;
            var slider = ABSKnob(key, cc);
            sliders.add(slider);
            ctls.put(key, slider);
        };
		masterValue.do {|cc, i|
            var key  = ("master").asSymbol;
            var knob = ABSKnob(key, cc);
            master.add(knob);
            ctls.put(key, knob);
        };

		soloValue.do {|note, i|
            var key = ("solo").asSymbol;
            var pad = ABSPad(key, note);
            solo.add(pad);
            ctls.put(key, pad);
        };

		bankLeftValue.do {|note, i|
            var key = ("bankLeft").asSymbol;
            var pad = ABSPad(key, note);
            bankLeft.add(pad);
            ctls.put(key, pad);
        };
		bankRightValue.do {|note, i|
            var key = ("bankRight").asSymbol;
            var pad = ABSPad(key, note);
            bankRight.add(pad);
            ctls.put(key, pad);
        };

        muteValues.collect {|note, i|
            var key = ("mutepad" ++ (i+1)).asSymbol;
            var pad = ABSPad(key, note);
            mutepads.add(pad);
            ctls.put(key, pad);
        };

		recValues.collect {|note, i|
            var key = ("recpad" ++ (i+1)).asSymbol;
            var pad = ABSPad(key, note);
            recpads.add(pad);
            ctls.put(key, pad);
        };
    }

    freeAll {
        ctls.do(_.free);
    }

    doesNotUnderstand {|selector ... args|
        ^ctls[selector] ?? { ^super.doesNotUnderstand(selector, args) }
    }
}

AkaiMPKMini {
var <knobs, <pads;

    var ctls;
    var knobValues, padValues;

    *new {
        ^super.new.init;
    }

    init {
        ctls = ();

        knobs = List[];
        pads  = List[];

        //                    Row1             Row2
	    knobValues = [ 7, 10, 8, 1,   12, 13, 11, 33];

		//                    Bank1 Row1      Bank1 Row2      Bank2 Row1       Bank2 Row2
        padValues  = [ 48, 49, 50, 51,  44, 45, 46, 19,  36, 37, 38, 39,  32,33,34,35  ];

        MIDIClient.init;
        MIDIIn.connectAll;

        this.assignCtls;
    }

    assignCtls {
        knobValues.do {|cc, i|
            var key  = ("knob" ++ (i+1)).asSymbol;
            var knob = ABSKnob(key, cc);
            knobs.add(knob);
            ctls.put(key, knob);
        };

        padValues.collect {|note, i|
            var key = ("pad" ++ (i+1)).asSymbol;
            var pad = ABSPad(key, note);
            pads.add(pad);
            ctls.put(key, pad);
        };
    }

    freeAll {
        ctls.do(_.free);
    }

    doesNotUnderstand {|selector ... args|
        ^ctls[selector] ?? { ^super.doesNotUnderstand(selector, args) }
    }
}

IRigPads {
    var <knobs, <pads;

    var ctls;
    var knobValues, padValues;

    *new {
        ^super.new.init;
    }

    init {
        ctls = ();

        knobs = List[];
        pads  = List[];

        knobValues = [ 10, 11, 1, 7 ];
        padValues  = [ 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63];

        MIDIClient.init;
        MIDIIn.connectAll;

        this.assignCtls;
    }

    assignCtls {
        knobValues.do {|cc, i|
            var key  = ("knob" ++ (i+1)).asSymbol;
            var knob = ABSKnob(key, cc);
            knobs.add(knob);
            ctls.put(key, knob);
        };

        padValues.collect {|note, i|
            var key = ("pad" ++ (i+1)).asSymbol;
            var pad = ABSPad(key, note);
            pads.add(pad);
            ctls.put(key, pad);
        };
    }

    freeAll {
        ctls.do(_.free);
    }

    doesNotUnderstand {|selector ... args|
        ^ctls[selector] ?? { ^super.doesNotUnderstand(selector, args) }
    }
}

ABSKnob {
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

ABSPad {
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
