import os
import time

class AudioServer:
    def start(self):
        raise NotImplementedError
    def set_amp(self):
        """ Amplitude or volume """
        raise NotImplementedError
    def set_pan(self):
        """ Panning value between -1..1 (Left / Right) """
        raise NotImplementedError
    def set_reverb_config(self, dry, wet):
        """ Reverb configuration dry/wet """
        raise NotImplementedError
    def set_reverb_dry(self, dry):
        raise NotImplementedError
    def set_reverb_wet(self, wet):
        raise NotImplementedError
    def vocoderplayfile(self):
        """ Simil vocoder effect """
        raise NotImplementedError

    freesound_desc_conv = {
        "content": "content",
        "bpm": "BPM",
        "duration": "sfx.duration",
        "inharmonicity.mean": "sfx.inharmonicity.mean",
        "hfc.mean": "lowlevel.hfc.mean",
        "spectral_centroid.mean": "lowlevel.spectral_centroid.mean",
        "spectral_complexity.mean": "lowlevel.spectral_complexity.mean"
    }
    def retrieve_new_sample(self, mir_state):

#         #FIXME: needs two descriptors? duration and other?
#         # duration always have a number (minor equal) < value
#         new_state["sfx.duration"] = "* TO %s"%new_state["sfx.duration"]
        print(mir_state.enabled)
        new_state = dict()
        for desc,enabled in mir_state.enabled.items():
            print("desc: %s"%desc)
            try:
                new_desc = self.freesound_desc_conv[desc] #TODO: make options for different api's (or reimplent in derived class)
            except:
                print("Falta key!") #FIXME
            if enabled==1:
                value = mir_state.desc[desc]
                sign = mir_state.sign[desc]
                if sign=="=":
                    new_state[new_desc] =  value
                elif sign=="<":
                    new_state[new_desc] =  "* TO %f"%value
                elif sign==">":
                    new_state[new_desc] =  "%f TO *"%value

#         #TODO: filter state values (search with AND only if they are enabled (on==1))
#         print("Estado MIR freesound a buscar: %s"%new_state)
        # new_state={"sfx.duration": "7.2",
        # "lowlevel.hfc.mean": "* TO 0.0005",
        # "lowlevel.spectral_complexity.mean": "1"
        # }
        print(new_state)

        file_chosen, author, sound_id  = self.api.get_one_by_mir(new_state)
#         #(needs to wait here?)
        
        print( os.path.getsize(file_chosen) )
        # time.sleep(3) #wait for ffmpeg conversion . FIXME: wait process completion in get_one_by_mir() method
        
        if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000:
            self.logging.debug(file_chosen+" by "+ author + " - id: "+str(sound_id)+"\n")
            self.vocoderplayfile(file_chosen)

#class