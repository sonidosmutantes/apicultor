s.reboot;
~files = FileReader.read("/home/miles/Musica/mp3list.txt",true,true,delimiter:$\n)[0].scramble;
~dur=2**15/44100;

//Método con synthdef y Buffer.cueSound
(
SynthDef("playfile", { arg bufnum = 0;
	var env = Env([0,1,0],[~dur/2.0,~dur/2.0],\welch);
    Out.ar(0, DiskIn.ar(2, bufnum))*EnvGen.kr(env, doneAction: 2);
}).load(s);
)

(
SynthDef("playfile2", {arg bufnum = 0;
	var env = Env([0,1,0],[~dur/2.0,~dur/2.0],\welch);
    Out.ar(0,
        PlayBuf.ar(2, bufnum, BufRateScale.kr(bufnum), doneAction:2)
    )*EnvGen.kr(env, doneAction: 2)
}).load(s)
)

~start = 2;

//Función para el Task
(
~read_and_play={
	m = MP3(~files[~files.size.rand]);
	m.start;
	b = Buffer.cueSoundFile(s, m.fifo, ~start*44100, 2,44100*~dur);
	x = Synth("playfile2", [\bufnum, b.bufnum], addAction:\addToTail);
	~dur.wait;
	b.free;
};
)

e = Task({ ~read_and_play.loop;0.5.wait; ~read_and_play.loop; });
e.start;
e.stop;

/*
b.close;b.free;m.finish;

//Método con readToBuffer y PlayBuf, con condition
(
~read_and_play={
x = Condition.new;
b = MP3.readToBuffer(s, ~files[~files.size.rand],0,44100*~dur,action: { x.test = true; x.signal });
{x.wait; {PlayBuf.ar(2,b)}.play; }.fork;
}
)

t = Task({ { ~read_and_play.fork; 2.0.wait;}.loop });
t.start;
t.stop;

bb.free;



/*
~pista = "/home/miles/Musica/Pez/2004\ -\ Pez\ -\ Folklore/14\ -\ Respeto.mp3"
~pista = "/home/miles/Musica/Bartok - Bluebeard's Castle/05 - Judith, open now the fourth door.mp3"

~pista = "/home/miles/test.mp3"

(
SynthDef("help_mp3_01", { |bufnum = 0|
    Out.ar(0, DiskIn.ar(2, bufnum));
}).load(s);
)

s.reboot;

m = MP3(~pista);
m.start;
// Now you can use it almost like any other file, by reading from m.fifo
b = Buffer.cueSoundFile(s, m.fifo, 0, 2);
x = Synth("help_mp3_01", [\bufnum, b.bufnum], addAction:\addToTail);
m.playing;
// You can stop and restart the piping (with a bit of a delay) - note what happens
m.stop;
m.playing;
m.start;
m.playing;
// Please remember to tidy up after yourself:
x.free;
b.close; b.free;
m.finish;

s.reboot;
m = MP3(~pista);
m.start;
b = Buffer.read(s, m.fifo, 0, 10000);
b.play;
b.close;b.free;
m.finish


TempoClock
f = File("/home/miles/Musica/mp3list.txt","r");
f.getLine(1024)
f.close;


*/