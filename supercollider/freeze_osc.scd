s.boot;

b = Buffer.read(s, "/Users/hordia/Documents/vmshared/samples/1194_sample1.wav");

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

SynthDef(\mFreezeD, { arg out=0, bufnum=0, point=0, vol=1, fftwidth=4096, pan=0, startP=0, endP=1;
    var trig, in, chain, startpos, endpos, sig;

    startpos = startP.linlin(0,1,0,BufRateScale.kr(bufnum));
	endpos = endP.linlin(0,1,0,BufRateScale.kr(bufnum));

    in = PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum), startpos,endpos ,loop:1 ); //funca con mouse

	chain = FFT(LocalBuf(2**11), in);
    chain = PV_MagFreeze(chain, point);

	sig = Splay.ar(vol * IFFT(chain).dup);
	sig = Balance2.ar(sig[0], sig[1], pan);
	Out.ar(out, sig);
}).add;

~channel = 0;
x = Synth(\mFreezeC, [\bufnum, b, \out, ~channel, \vol, 0.2, \point, 1]);
//x = Synth(\mFreezeD, [\bufnum, b, \out, ~channel, \vol, 0.2, \point, 1]);
x.free;

x.set(\vol, 1.9);
x.set(\vol, 0.5);
x.set(\point, 1);
x.set(\point, 0);
//x.get(\point, { arg val; val.postln} );

x.set(\startp, 1);
x.get(\startp, { arg val; val.postln} );
//x.set(\startP, 0.5);

//OSC
NetAddr.langPort;

OSCdef.new(
	\toggleFreeze,
	{
		arg msg;
		x.set(\point, msg[1].linlin(0,127,0,1));
	},
	'/solo7',
);

OSCdef.new(
	\changeVolume,
	{
		arg msg;
		x.set(\vol, msg[1].linlin(0,1,0,2));
	},
	'/vol7',
);

OSCdef.new(
	\changeStartPos,
	{
		arg msg;
		x.set(\startP, msg[1].linlin(0,1,0,1));
	},
	'/vol8',
);

OSCdef.new(
	\changeEndPos,
	{
		arg msg;
		x.set(\endP, msg[1].linlin(0,1,-1,1));
	},
	'/vol9',
);

OSCdef.new(
	\changePan,
	{
		arg msg;
		x.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/vol10',
);

