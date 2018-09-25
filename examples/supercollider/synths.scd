
//Play synth
SynthDef(\playBufMono, {| out = 0, bufnum = 0, vol=1, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, vol * player).dup }).add;

SynthDef(\playBufMonoM, {| out = 0, bufnum = 0, vol=1, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, vol * player) }).add;

// Freeze
SynthDef(\mutantefreeze, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startPos=0;
    var in, chain, sig;
    in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum),loop: 1);
	chain = FFT(LocalBuf(4096), in);
    chain = PV_MagFreeze(chain, point);

	//with PAN
	sig = Splay.ar(vol * IFFT(chain).dup);
	sig = Balance2.ar(sig[0], sig[1], pan);
	Out.ar(out, sig);
}).add;

//startP is the number of the sample to start from
SynthDef(\mFreezeC, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startP=0.5;
    var trig, in, chain, startpos, sig;

	trig = Impulse.kr( startP.linlin( 0,1, 1, 100) );
    startpos = startP.linlin(0,1,0,BufRateScale.kr(bufnum));

    in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum), trig,startpos ,loop:1 ); //funca con mouse

	chain = FFT(LocalBuf(2**11), in);
    chain = PV_MagFreeze(chain, point);

	sig = Splay.ar(vol * IFFT(chain).dup);
	sig = Balance2.ar(sig[0], sig[1], pan);
	Out.ar(out, sig);
	//Out.ar(out, vol * IFFT(chain).dup);
}).add;


~channel = 0; //~speaker0
// ~channel = 1;
~rate = 1; //normal
// ~rate = 0.5; //half speed

x.set(\vol,2);

x = Synth(\mutantefreeze, [\bufnum, ~bank1a, \out, ~channel, \vol, 0]);
//x.free;
//alt ->freeze Rate
//x = Synth(\mutantefreezeRate, [\bufnum, ~bank1a, \out, ~channel, \vol, 0, \rate, 0.1]);

y = Synth(\mutantefreeze, [\bufnum, ~bank1b, \out, ~channel, \vol, 0]);
y.free;
y = Synth(\mFreezeC, [\bufnum, ~bank1b, \out, ~channel, \vol, 0]);

z = Synth(\mutantefreeze, [\bufnum, ~bank1c, \out, ~channel, \vol, 0]);
q = Synth(\mutantefreeze, [\bufnum, ~bank1d, \out, ~channel, \vol, 0]);

g = Synth(\mutantefreeze, [\bufnum, ~bank2a, \out, ~channel, \vol, 0]);
h = Synth(\mutantefreeze, [\bufnum, ~bank2b, \out, ~channel, \vol, 0]);
i = Synth(\mutantefreeze, [\bufnum, ~bank2c, \out, ~channel, \vol, 0]);
j = Synth(\mutantefreeze, [\bufnum, ~bank2d, \out, ~channel, \vol, 0]);

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

(
SynthDef(\phaser, { arg out=0, in=0;

    var input,dsig, mixed;
    input = SoundIn.ar(in, 1);

    dsig = AllpassL.ar(input, 4, SinOsc.ar(2, 0, 0.005, 0.005), 0);
    mixed = input + dsig;
    Out.ar([out, out+1], mixed);
}).load(s);
)

~channel = 0; //~speaker0
//a = Synth(\phaser, addAction:\addToTail)
a = Synth(\phaser, [\in, ~bank1b, \out, ~channel])

