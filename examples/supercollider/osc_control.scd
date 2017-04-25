
NetAddr.langPort; //osc port 57120 ó 57121

//ver todos los mensajes
OSCFunc.trace
OSCFunc.trace(false); // Turn posting off

// Pan control
OSCdef.new(
	\pan1A,
	{
		arg msg;
		x.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/pan2', //bug manda pan2 en lugar de pan3
);

OSCdef.new(
	\pan2A,
	{
		arg msg;
		y.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/pan4',
);

OSCdef.new(
	\pan2C,
	{
		arg msg;
		z.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/pan5',
);

OSCdef.new(
	\pan2D,
	{
		arg msg;
		q.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/pan6',
);

//control vol / bank 1 A
~maxvol = 7
OSCdef.new(
	\vol1A,
	{
		arg msg;
		x.set(\vol, msg[1].linlin(0,1,0,~maxvol));
	},
	'/vol7',
);

OSCdef.new(
	\pan1A,
	{
		arg msg;
		x.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/pan6', //bug en Control App (en lugar de /pan7 manda pan6)
);

//control PAN+time / bank 1 B
OSCdef.new(
	\pan1B,
	{
		arg msg;
		y.set(\pan, msg[1].linlin(0,1,-1,1));
	},
	'/pan8', //bug en Control App (en lugar de /pan7 manda pan6)
);

OSCdef.new(
	\time1B,
	{
		arg msg;
		y.set(\startP, msg[1].linlin(0,1,0,1));
	},
	'/vol8',
);

~maxvol = 10
OSCdef.new(
	\vol1B,
	{
		arg msg;
		y.set(\vol, msg[1].linlin(0,1,0,~maxvol));
	},
	'/vol9',
);
