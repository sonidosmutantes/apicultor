//Address and port
NetAddr.localAddr //check '127.0.0.1', 57120

s.boot; //start server

//create the buffer
b = Buffer.alloc(s, s.sampleRate * 8.0, 2); // an 8 second stereo buffer

//play
SynthDef(\playBufMono,
	{| out = 0, bufnum = 0, vol=1, rate = 1 |

	var scaledRate, player;
	scaledRate = rate * BufRateScale.kr(bufnum);
	player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);
	Out.ar(out, vol * player).dup }
).add;


//freeze
SynthDef(\mutantefreeze, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startPos=0;
    var in, chain, sig;
    in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
	chain = FFT(LocalBuf(4096), in);
    chain = PV_MagFreeze(chain, point);

	//Out.ar(out, vol * IFFT(chain).dup);

	//with PAN
	sig = Splay.ar(vol * IFFT(chain).dup);
	sig = Balance2.ar(sig[0], sig[1], pan);
	Out.ar(out, sig);
}).add;

//Playing resource
x = OSCFunc(
	{ | msg, time, addr, port |
		var pyFile;

		~new_file = msg[1].asString;
		b = Buffer.read(s, ~new_file);
		( "New sound received " + pyFile ).postln;

		l = Synth(\playBufMono, [\out, 0, \bufnum, b.bufnum, \rate, 1]);
		r = Synth(\playBufMono, [\out, 1, \bufnum, b.bufnum, \rate, 1]);
		},

	    '/play'
);

x = OSCFunc(
	{ | msg, time, addr, port |
		var pyFile;

		~new_file = msg[1].asString;
		b = Buffer.read(s, ~new_file);
		( "New sound received " + pyFile ).postln;

		~tmp_mutante_freeze = Synth(\mutantefreeze, [\bufnum, b.bufnum, \out, 0, \vol, 1]);
		},

	    '/playfreeze'
);


~tmp_mutante_freeze.set(\point, 0); //freeze off
~tmp_mutante_freeze.set(\point, 1); //freeze on

x.free; //remove the osc function

s.quit; //stops server

------------------

//Oscillator resource
( SynthDef( \sin, { | amp = 0.01, freq = 333, trig = 1 | var env, sig; env = EnvGen.kr( Env.asr( 0.001, 0.9, 0.001 ), trig, doneAction: 0 ); sig = LFTri.ar( [ freq, freq * 0.999 ], 0.0, amp ) * env; Out.ar( [ 0 ], sig * 0.6 ); }).add; h = Synth( \sin, [ \amp, 0.4 ] ); x = OSCFunc( { | msg, time, addr, port | var pyFreq; pyFreq = msg[1].asFloat; ( "freq is " + pyFreq ).postln; h.set( \freq, pyFreq ); }, '/osc' ); )
