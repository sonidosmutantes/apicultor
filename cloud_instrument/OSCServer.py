import liblo
from liblo import make_method

class OSCServer(liblo.ServerThread):
    """
        OSC server
    """
    audio_server = None #TODO: build set_up method
    state = None #TODO: build set_up method
    
    def __init__(self, port):
        liblo.ServerThread.__init__(self, port)
        self.debug = True #Received osc msg to std output
    #()

    #General FX
    @make_method("/fx/volume", 'f')
    def update_volume_callback(self, path, args):
        value = args[0]
        print("Update volume: %f"%value)
        self.audio_server.set_amp(float(value)) # server amplitude

    @make_method("/fx/pan", 'f')
    def update_pan_callback(self, path, args):
        value = args[0]
        print("Update PAN: %f"%value)
        self.audio_server.set_pan(float(value)) # server panning
    
    @make_method("/fx/reverb", 'ff')
    def update_reverb_callback(self, path, args):
        dry = float(args[0])
        wet = float(args[1])
        print("Update reverb. Dry: %f. Wet: %f"%(dry,wet))
        self.audio_server.set_reverb_config(dry,wet)

    @make_method("/fx/pitch", 'ff')
    def pitch_shift_callback(path, args):
        raise NotImplementedError
    # print("Received %s %s"%(path,args))
    # try:
    #     value = args[0]
    #     state = int(args[1])
    #     if state==0: return #on release
    #     #take a reference (now 60 TODO update) and calculate amount of steps shift
    #     amount = pow(2,(value-60.)/12.)
    #     # FIXME
    #     print("Pitch shift amount %f"%amount)
    #     c = FreqShift(c, shift=amount, mul=volume)
    # except Exception, e:
    #     print(e)

    @make_method("/fx/reverbdry", 'f')
    def update_reverbdry_callback(self, path, args):
        dry = float(args[0])
        print("Update reverb. Dry: %f"%dry)
        self.audio_server.set_reverb_dry(dry)

    @make_method("/fx/reverbwet", 'f')
    def update_reverbwet_callback(self, path, args):
        wet = float(args[0])
        print("Update reverb. Wet: %f"%wet)
        self.audio_server.set_reverb_wet(wet)

    @make_method("/set_voices", 'ff')
    def update_set_voices_callback(self, path, args):
        voice = int(args[0])
        status = int(args[1])
        if status==1:
            print("Update Voice %i. Status: %i"%(voice,status))
            self.audio_server.set_enabled_voice(voice)

    @make_method("/retrieve", 'i')
    def retrieve_button_callback(self, path, args):
        state = args[0]
        if self.debug:
            print("Retrieve status: %i"%state)
        if state==1:
            self.audio_server.retrieve_new_sample( self.state )

    #General chat messages
    @make_method("/chat", 's')
    def chat_callback(self, path, args):
        print("Chat message: %s"%args[0])
    #()

    #MIR descriptors methods
    @make_method(None, 'f') # /mir/*
    def mir_fallback(self, path, args):
        # print("received message '%s'" % path)
        mir = path[len('/mir/'):].replace('/','.').lower()
        value = args[0]
        if self.debug:
            print("Update %s: %f"%(path,value))
            print("mir: %s"%mir)
        self.state.set_desc(mir, value)
    #()

    # @make_method(None, None)
    # def fallback(self, path, args):
    #     print("Received unknown message '%s'" % path)
    # #()

    @make_method(None, None)
    def update_state_fallback(self, path, args, types, src):
        # print("got unknown message '%s' from '%s'" % (path, src.url))
        # for a, t in zip(args, types):
        #     print("argument of type '%s': %s" % (t, a))
        msg = path[1:]
        value = args[0]
        print("Received %s %s"%(path,args))
    #()

#class