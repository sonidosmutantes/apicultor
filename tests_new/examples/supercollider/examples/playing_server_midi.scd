//Address and port
NetAddr.localAddr //check '127.0.0.1', 57120

s.boot; //start server

//create the buffer
b = Buffer.alloc(s, s.sampleRate * 8.0, 2); // an 8 second stereo buffer

SynthDef(\playBufMono, {| out = 0, bufnum = 0, rate = 1 |  var scaledRate, player;
scaledRate = rate * BufRateScale.kr(bufnum);  player = PlayBuf.ar(1, bufnum,scaledRate, doneAction:2);  Out.ar(out, player) }).add;

//WGET AND OBTAIN VALUE
//"wget http://127.0.0.1:5000/pistas/126/descriptor/lowlevel.hfc.mean -O desc.tmp".unixCmd;
//f = FileReader.read("./desc.tmp".standardizePath);

(
var notes, on, off;
var pyFile;

MIDIClient.init;
MIDIIn.connectAll;

notes = Array.newClear(128);    // array has one slot per possible MIDI note


pyFile = "/Users/hordia/Documents/apicultor/samples/126_sample3.wav";
b = Buffer.read(s,pyFile);

on = MIDIFunc.noteOn({ |veloc, num, chan, src|
    notes[num] = Synth(\playBufMono, [\out, 1, \bufnum, b.bufnum, \rate, 1]); ( "New note received " + num ).postln;
});

off = MIDIFunc.noteOff({ |veloc, num, chan, src|
    notes[num].release;
});

q = { on.free; off.free; };
)

//Playing resource
//x = OSCFunc( { | msg, time, addr, port | var pyFile; pyFile = msg[1].asString; ( "New sound received " + pyFile ).postln; l = Synth(\playBufMono, [\out, 0, \bufnum, b.bufnum, \rate, 1]); r = Synth(\playBufMono, [\out, 1, \bufnum, b.bufnum, \rate, 1]); b = Buffer.read(s,pyFile); }, '/play' );
//x.free; //remove the osc function

s.quit; //stops server