	//Surprise sound (TODO: add MIR parameter)
		m = 15; //Length of sound list (TODO: retrieve from API) FIXME

//Get a new sample file from apicultor (HFC < 1 )
		format("curl http://%:5000/search/mir/samples/HFC/lessthan/1000/% -o /Users/hordia/desc.tmp", ~ip,m).unixCmd; //mac os

//Search MIR desc greater than (SAMPLES) HFC>40 / 5 valores
		//format("curl http://apicultor:5000/search/mir/samples/HFC/greaterthan/40000/% -o /Users/hordia/desc.tmp", ~ip,m).unixCmd; //mac os

//duration <1
		//format("curl http://apicultor:5000/search/mir/samples/duration/lessthan/1000/% -o /Users/hordia/desc.tmp", ~ip,m).unixCmd; //mac os

//duration >2s
		//format("curl http://apicultor:5000/search/mir/samples/duration/greaterthan/2000/% -o /Users/hordia/desc.tmp", ~ip,m).unixCmd; //mac os

		//FIXME: wait to download here? (takes effect next time)

		f = FileReader.read("/Users/hordia/desc.tmp".standardizePath); //array
        v = f.at(m.rand)[0]; //select a random value from array (0..10 range)
        v.postln(); //selected file
        f = ("/Users/hordia/Documents/vmshared"+v.replace("./","/")).replace(" ",""); //trim spaces (TODO: check why there is an extra space in the path)

        a = Buffer.read(s, f ); // new buffer A

		//plays new sample (two channels)
		//w.free;
		//w = ~prepare_freeze.value(~bank1a, ~channel, "Bank1", "A");
        x = Synth(\mutantefreeze, [\bufnum, a, \out, ~channel, \vol, 0]);
		        //plays new sample
				//plays new sample
		        ~speaker1 = 0;
				~speaker2 = 1;
		        //r = Synth(\playBufMono, [\out, ~speaker0, \bufnum, a.bufnum, \rate, 1]); //e @ L channel
				r = Synth(\playBufMono, [\out, ~speaker1, \bufnum, a, \rate, 1]); //e @ R channel
x.free;
r.free;