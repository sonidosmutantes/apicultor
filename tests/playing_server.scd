//Address and port
NetAddr.localAddr //check '127.0.0.1', 57120

s.boot; //start server

SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, player) }).add;

//Playing resource
x = OSCFunc( { | msg, time, addr, port | var pyFile; pyFile = msg[1].asString; ( "New sound received " + pyFile ).postln; l = Synth(\playBufMono, [\out, 0, \bufnum, b.bufnum, \rate, 1]); r = Synth(\playBufMono, [\out, 1, \bufnum, b.bufnum, \rate, 1]); b = Buffer.read(s,pyFile); }, '/play' );

x.free; //remove the osc function

s.quit; //stops server

------------------

//Oscillator resource
( SynthDef( \sin, { | amp = 0.01, freq = 333, trig = 1 | var env, sig; env = EnvGen.kr( Env.asr( 0.001, 0.9, 0.001 ), trig, doneAction: 0 ); sig = LFTri.ar( [ freq, freq * 0.999 ], 0.0, amp ) * env; Out.ar( [ 0 ], sig * 0.6 ); }).add; h = Synth( \sin, [ \amp, 0.4 ] ); x = OSCFunc( { | msg, time, addr, port | var pyFreq; pyFreq = msg[1].asFloat; ( "freq is " + pyFreq ).postln; h.set( \freq, pyFreq ); }, '/osc' ); )
