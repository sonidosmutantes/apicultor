import platform
from pyo import *
from AudioServer import AudioServer

class PyoAudioServer(AudioServer):
    # volume = 0.
    # pan = 0.
    # dry = 1.
    # wet = 0.
    pyo_server = None

    def __init__(self):
        # TODO: implement MIDI support        
        # notes = Notein(poly=10, scale=1, mul=.5)
        # p = Port(notes['velocity'], .001, .5)
        pass
    #()

    def start(self):
        """ Pyo sound server """
        if platform.system() == "Darwin" or platform.system() == "Windows":
            ### Default
            s = Server().boot()
            # s = Server(duplex=0).boot()
            # s = Server(audio='portaudio').boot()
            s = Server().boot()
        else: #Linux
            ### JACK ###
            # or export PYO_SERVER_AUDIO=jack (~/.bashrc)
            s = Server(audio='jack')
            s.setJackAuto(False, False) #some linux bug workaround
            s.boot()
            s.setJackAutoConnectOutputPorts(['system:playback_1', 'system:playback_2'])

        s.start() #no GUI s.gui(locals())

        # sffade = Fader(fadein=0.05, fadeout=1, dur=0, mul=0.5).play()

        # Mixer
        # 3 outputs mixer, 1 second of amplitude fade time
        #mm = Mixer(outs=3, chnls=2, time=1)

        self.dry_val = 1  #0.7
        self.wet_val = 0.5 # 0.3, check which reverb algorithm is using

        a = Sine(freq=10, mul=0.3) #start signal

        self.c = Clip(a, mul=self.dry_val)

        #Dry Output to sound card
        self.out = self.c.mix(2).out() #dry output
        
        # Reverb
        # b1 = Allpass(out, delay=[.0204,.02011], feedback=0.35) # wet output
        # b2 = Allpass(b1, delay=[.06653,.06641], feedback=0.41)
        # b3 = Allpass(b2, delay=[.035007,.03504], feedback=0.5)
        # b4 = Allpass(b3, delay=[.023021 ,.022987], feedback=0.65)
        # c1 = Tone(b1, 5000, mul=0.2).out()
        # c2 = Tone(b2, 3000, mul=0.2).out()
        # c3 = Tone(b3, 1500, mul=0.2).out()
        # c4 = Tone(b4, 500, mul=0.2).out()

        #Another reverb
        # comb1 = Delay(out, delay=[0.0297,0.0277], feedback=0.65)
        # comb2 = Delay(out, delay=[0.0371,0.0393], feedback=0.51)
        # comb3 = Delay(out, delay=[0.0411,0.0409], feedback=0.5)
        # comb4 = Delay(out, delay=[0.0137,0.0155], feedback=0.73)
        # combsum = out + comb1 + comb2 + comb3 + comb4
        # all1 = Allpass(combsum, delay=[.005,.00507], feedback=0.75)
        # all2 = Allpass(all1, delay=[.0117,.0123], feedback=0.61)
        # lowp = Tone(all2, freq=3500, mul=self.wet_val).out()

        #Reverb
        """
        8 delay lines FDN (Feedback Delay Network) reverb, with feedback matrix based upon physical modeling scattering junction of 8 lossless waveguides of equal characteristic impedance.
        """
        pan = SPan(self.out, pan=[.25, .4, .6, .75]).mix(2)
        self.reverb = WGVerb(pan, feedback=.65, cutoff=3500, bal=.2)
        # rev.out()

        #Gate
        gt = Gate(self.reverb, thresh=-24, risetime=0.005, falltime=0.01, lookahead=5, mul=.4)
        gt.out() #wet ouput? 


        # Loads the sound file in RAM. Beginning and ending points
        # can be controlled with "start" and "stop" arguments.
        # t = SndTable(path)

        self.pyo_server = s
        return s # returns audio server instance
    #start()

    def set_amp(self, new_value):
        self.pyo_server.amp = float(new_value)
    #()
    def set_pan(self, new_value):
        self.pyo_server.pan = float(new_value)
    #()
    def set_reverb_config(self, dry=1., wet=0.):
        # self.set_reverb_dry(dry)
        # self.set_reverb_wet(wet)
        balance = dry-wet
        self.reverb.setBal(balance)
    #()
    def set_reverb_dry(self, dry):
        self.dry_val = dry
        # self.c.setInput(pvs, fadetime=.25)
        balance = self.wet_val - self.dry_val
        self.reverb.setBal(balance)
    #()
    def set_reverb_wet(self, wet):
        self.wet_val = wet
        #FIXME falta aplicar wet al proceso correspondiente
        balance = self.wet_val - self.dry_val
        self.reverb.setBal(balance)
    #()

    # def freesound_search(api_key="", id=""):
    #     call = """curl -H "Authorization: Token %(api_key)s" 'http://www.freesound.org/apiv2/sounds/%(id)s/'"""%locals()
    #     response = urllib2.urlopen(call).read()
    #     print(response)
    # #freesound_search()


    def vocoderplayfile(self, new_file, dry_value=1., loop_status=True):
        """
            default synth (freeze intent)
            vocoder schema
        """

        #TODO: pending normalizar file audio input (luego de convertir a wav)

        #Phase Vocoder
        sfplay = SfPlayer(new_file, loop=loop_status, mul=dry_value)
        pva = PVAnal(sfplay, size=1024, overlaps=4, wintype=2)
        pvs = PVAddSynth(pva, pitch=1., num=500, first=10, inc=10).mix(2)#.out() 
        # pvs = PVAddSynth(pva, pitch=notes['pitch'], num=500, first=10, inc=10, mul=p).mix(2).out()

        self.c.setInput(pvs, fadetime=.25)
        print("setInput new: %s"%new_file)
        self.out = self.c.mix(2).out() #dry output

        # TODO: apply fx chain again
    #playfile()

    def pyo_synth_noisevc(self, new_file, dry_value):
        print("noise vocoder synth")
        # First sound - dynamic spectrum.
        spktrm = SfPlayer(new_file, speed=[1,1.001], loop=True, mul=dry_value)

        # Second sound - rich and stable spectrum.
        excite = Noise(0.2)

        # LFOs to modulated every parameters of the Vocoder object.
        lf1 = Sine(freq=0.1, phase=random()).range(60, 100)
        lf2 = Sine(freq=0.11, phase=random()).range(1.05, 1.5)
        lf3 = Sine(freq=0.07, phase=random()).range(1, 20)
        lf4 = Sine(freq=0.06, phase=random()).range(0.01, 0.99)

        voc = Vocoder(spktrm, excite, freq=lf1, spread=lf2, q=lf3, slope=lf4, stages=32)

        self.c.setInput(voc, fadetime=.25)
        # c = c.mix(2).out()
    #pyo_synth_noisevc()

    def granular_synth(self, new_file):
        """
            Granulator sound
        """
        snd = SndTable(file_chosen)
        env = HannTable()
        # note_in_pitch = 62
        # posx = Port( Midictl(ctlnumber=[78], minscale=0, maxscale=snd.getSize()), 0.02)
        # posf = Port( Midictl(ctlnumber=[16], minscale=0, maxscale=snd.getSize()), 0.02)
        #porta = Midictl(ctlnumber=[79], minscale=0., maxscale=60.)
        # posxx = (note_in_pitch-48.)/(96.-48.)*posf+posx
        # pos = SigTo(posxx)
        # tf = TrigFunc(Change(porta), function=set_ramp_time)
        # pitch = Port(Midictl(ctlnumber=[17], minscale=0.0, maxscale=2.0),0.02)
        # noisemul = Midictl(ctlnumber=[18], minscale=0.0, maxscale=0.2)
        # noiseadd = Port(Midictl(ctlnumber=[19], minscale=0.0, maxscale=1.0),0.02)
        # dur = Noise(mul=noisemul)+noiseadd
        pitch = 62
        dur = 3
        pos = 1
        g = Granulator(snd, env, pitch*0.1/dur, pos , dur, 16, mul=.3).mix(2).out()
    #granulator_synth()

    #TODO: chequear si se usa
    # def set_ramp_time():
    #     pos.time = porta.get()

    def playfile(self, new_file, dry_value=1., loop_status=True):
        self.vocoderplayfile(new_file)
    #()
#class