b = Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "/../data/126.wav");

(
//trig with MouseY
SynthDef("magFreeze", { arg out=0, soundBufnum=2;
    var in, chain;
    in = PlayBuf.ar(1, soundBufnum, BufRateScale.kr(soundBufnum),loop: 1);
    chain = FFT(LocalBuf(4096), in);
    chain = PV_MagFreeze(chain, MouseY.kr>0.5);
    Out.ar(out, 0.1 * IFFT(chain).dup);
}).add;
)

x = Synth("magFreeze",[\out,0,\soundBufnum, b]);

x.free;




