/*
    Sounds from the web (using API-Cultor) by hordia

    live-coding
*/


/*
// control sending OSC to (own) supercollider service
//OSC address and port
//NetAddr.localAddr //check '127.0.0.1', 57120
~netaddr = NetAddr("127.0.0.1", 57120)
~voice_number = 0; // [0..7]
~hardcoded_file = "/Users/hordia/samples/freesound-sessions/BREATHE_IN_2.wav";
~netaddr.sendMsg("/playfreeze", ~hardcoded_file,~voice_number); // (method, file_path, voice_number)
*/

/*
*/
// Control sending OSC to apicultor server (https://github.com/sonidosmutantes/apicultor)
// cloud_instrument$ python CloudInstrument.py
~apicultorsrv = NetAddr("127.0.0.1", 9001);

~voice = 4; // [1..8]
~apicultorsrv.sendMsg("/set_voices", ~voice, 1); //voice 2, en on (1)
~apicultorsrv.sendMsg("/mir/spectral_centroid/mean", 100);
~apicultorsrv.sendMsg("/mir/spectral_complexity/mean", 0.4);
~apicultorsrv.sendMsg("/mir/duration", 3.4); //tip: misma duración, /mantengo tempo/
~apicultorsrv.sendMsg("/mir/pitch_centroid/mean", 0.28);
~apicultorsrv.sendMsg("/retrieve", 1); //get a new sound

~apicultorsrv.sendMsg("/mir/duration", 10); // <10
~apicultorsrv.sendMsg("/mir/BPM", 110);

//controls and effects
~voice = 2;
//freeze [on/off]
~synthInstances[~voice-1].set(\point, 1); //on
~synthInstances[~voice-1].set(\point, 0); //off
*/


//task duration yield (volume!)
(
    var dur;
    var max = 4;
    var amount = 100;
    dur = Pseq([max, max/2, .8, 0.5, 0.5, .3, 1, max, max/2, max/3], amount).asStream;

    r = Task({
	var delta;
	while {
	    	delta = dur.next;
    		delta.notNil
	} {
    		"Duration: "++delta.postln;
    		~synthInstances[~voice-1].set(\vol, delta);
    		delta.yield;
	}
    }).play(quant: TempoClock.default.beats + 1.0);
)

//e = Env([800,300,400,500,200], [1,1,1,1], \exp); // 4 segments
//e = Env([0,-1,-0.5,0,1], [1,1,1,1], \sine).plot; // sine
//e = Env.new([0,1,0], [0.5, 1]).plot;

//pan control
(
		// reset then play again
        e = Env([0,-1,-0.5,0,1], [1,1,1,1], \sine);
		~synthInstances[~voice-1].set(\pan, e.asArray);
)

// granular synth
//~bufferinstances[~voice].bufnum.postln;
~voice = 2;
{
		100.do{ arg j;
			var timeprop = (j/199.0)**3;
			Synth(\sfgrain,[\bufnum, ~bufferinstances[~voice].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1), \pan, 1.0.rand2]);
			rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
		};
}.fork



//task granular trigger
(
    var dur;
    var max = 10;
    var amount = 1;
    dur = Pseq([max, max/2, max/5, 0, 0, max/4, max/3, max, max/2, max/3], amount).asStream;

    r = Task({
	var delta;
	while {
	    	delta = dur.next;
    		delta.notNil
	} {
    		"Duration: "++delta.postln;
    		//~synthInstances[~voice-1].set(\vol, delta);
		{
			100.do{ arg j;
			var timeprop = (j/199.0)**3;
			Synth(\sfgrain,[\bufnum, ~bufferinstances[~voice].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1), \pan, 1.0.rand2]);
			rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
			};
		}.fork;

    		delta.yield;
	}
    }).play(quant: TempoClock.default.beats + 1.0);
)


/*
//pre-perfo
~v1 = "/Users/hordia/samples/freesound-sessions/BREATHE_IN_2.wav";
//TODO: build in one line with a script
~netaddr.sendMsg("/playfreeze", ~v1, 0);
~netaddr.sendMsg("/playfreeze", ~v2, 1);
~netaddr.sendMsg("/playfreeze", ~v3, 2);
~netaddr.sendMsg("/playfreeze", ~v4, 3);
~netaddr.sendMsg("/playfreeze", ~v5, 4);
~netaddr.sendMsg("/playfreeze", ~v6, 5);
~netaddr.sendMsg("/playfreeze", ~v7, 6);
~netaddr.sendMsg("/playfreeze", ~v8, 7);
*/

/*
Voice with explanation about API-Culture and sound from the Cloud...
*/
~mir_voice  = Buffer.read(s, "/Users/hordia/samples/speech-sounds-from-the-cloud.wav");
~voice = Synth(\mutantefreeze, [\bufnum, ~mir_voice.bufnum, \out, 0, \vol, 1])
~voice.set(\point, 0); //off
~voice.set(\point, 1); //on
~voice.set(\vol, 0);

~surprise = Buffer.read(s, "/Users/hordia/samples/diciembre-bici.wav");
~surprise_synth = Synth(\mutantefreeze, [\bufnum, ~surprise.bufnum, \out, 0, \vol, 1])
~surprise_synth.set(\point, 0); //off
~surprise_synth.set(\point, 1); //on
~surprise_synth.set(\vol, 0);

//task duration yield (volume!)
(
    var dur;
    var max = 2;
    var amount = 1;
    dur = Pseq([max, max/2, 0.8, 0.5, 0.5, 0.3, 1, max, max/2, max/3], amount).asStream;

    r = Task({
	var delta;
	while {
	    	delta = dur.next;
    		delta.notNil
	} {
    		"Duration: "++delta.postln;
    		~surprise_synth.set(\vol, delta);
    		delta.yield;
	}
    }).play(quant: TempoClock.default.beats + 1.0);
)