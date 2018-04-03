/*
    Sounds from the web (using API-Cultor) by hordia

    live-coding
*/

//TODO:
/*
//NORMALIZAR VOLUMENES  de los samples --> para poder usar síntesis granular y freeze al mismo tiempo

+ patrones para disparar volumenes y sintesis granular
++ efectos en vivo (ej pitch shift vía pattern?)

arreglar paneo con pattern (no funca creo)
 --->  usb tracktor main output (+ headphones)
TODO: revisar traer sonidos por BPM fijo y usar time-stretch para cambiar
ritmos ccon tidal, traer sample.Y ajustar en el momento, luego dar el trigger
shaders kodelife --> tiene audio rítmico por mic (3 secciones por umbral, ajustable el bw)

a veces no activa el canal? 5, 6, nunca los pude manejar (esos canales, etc9


retrieve sounds using the note as descriptor? db supports it? c5, a5, etc
(ver)

copy first two channels to 3 and 4 (cuadraphonic duplicate)


Cargar samples en directorio para superdirt y manejar desde Tidal
       segmentar cada archivo en varias variantes del mismo sampler
usar varios sonidos del mismo cluster (para cada d1 $ etc)
--> ver lo que quedó grabado... (escuchar)
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

~voice = 5; // [1..8]
~apicultorsrv.sendMsg("/set_voices", ~voice, 1); //voice 2, en on (1)

~apicultorsrv.sendMsg("/mir/spectral_centroid/mean", 500);
~apicultorsrv.sendMsg("/mir/spectral_complexity/mean", 0.8);
~apicultorsrv.sendMsg("/mir/duration", 5); //tip: misma duración, /mantengo tempo/
~apicultorsrv.sendMsg("/mir/pitch_centroid/mean", 0.1);
~apicultorsrv.sendMsg("/retrieve", 1); //get a new sound

~apicultorsrv.sendMsg("/mir/duration", 10); // <10
~apicultorsrv.sendMsg("/mir/BPM", 110);

//controls and effects
~voice = 1;
//freeze [on/off]
~synthInstances[~voice-1].set(\point, 1); //on
~synthInstances[~voice-1].set(\point, 0); //off

//
~synthInstances[~voice-1].set(\pan, 0.9); //PAN
~synthInstances[~voice-1].set(\vol, 0); //vol
*/

//~synthInstances[~voice-1].set(\vol, 0);

//task duration yield (volume!)
(
    var dur;
    var max = 1;
    var amount = 100;
   // dur = Pseq([max, max/2, 0.8, 0.5, 0.5, 0.3, 1, max, max/2, max/3], amount).asStream;
   dur = Pseq([1, 0.5, 0.8, 0.5, 0.5, 0.3, 1, 1, 2, 0.4], amount).asStream;

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

//este no . ver el de abajo
~voice = 1;
{
		100.do{ arg j;
			var timeprop = (j/199.0)**3;
			Synth(\sfgrain,[\bufnum, ~bufferinstances[~voice].bufnum, \startPos,rrand(0.0,timeprop),\amp, exprand(0.005,0.1), \pan, 1.0.rand2]);
			rrand((timeprop*0.1).max(0.01),timeprop*0.4).wait
		};
}.fork


~voice = 4;
//task granular trigger
(
    var dur;
    var max = 10;
    var amount = 100;
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
~voice.set(\vol, 0); //TODO: fade out

~surprise = Buffer.read(s, "/Users/hordia/samples/diciembre-bici.wav");
~surprise_synth = Synth(\mutantefreeze, [\bufnum, ~surprise.bufnum, \out, 0, \vol, 0])
~surprise_synth.set(\point, 0); //off
~surprise_synth.set(\point, 1); //on
~surprise_synth.set(\vol, 0.3);

~voice = 4;
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
