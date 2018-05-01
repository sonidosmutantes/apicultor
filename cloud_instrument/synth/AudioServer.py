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
    def playfile(self, new_file, metadata, dry_value=1., loop_status=True):
        raise NotImplementedError
    def errorfile(self, metadata=""):
        raise NotImplementedError
    def notify_searching(self, msg="Hi"):
        raise NotImplementedError

    freesound_desc_conv = {
        "content": "content",
        "bpm": "rhythm.bpm",
        "duration": "sfx.duration",
        "inharmonicity.mean": "sfx.inharmonicity.mean",
        "hfc.mean": "lowlevel.hfc.mean",
        "pitch.mean": "lowlevel.pitch.mean",
        "pitch_centroid.mean": "sfx.pitch_centroid.mean",
        "spectral_centroid.mean": "lowlevel.spectral_centroid.mean",
        "spectral_complexity.mean": "lowlevel.spectral_complexity.mean"
    }

    def set_enabled_voice(self, voice):
        self.enabled_voice = voice
    #()

    def retrieve_new_sample(self, mir_state):
#         #FIXME: needs two descriptors? duration and other?
#         # duration always have a number (minor equal) < value
#         new_state["sfx.duration"] = "* TO %s"%new_state["sfx.duration"]
        print(mir_state.enabled)
        self.notify_searching("Searching in the Cloud...")
        new_state = dict()
        for desc,enabled in mir_state.enabled.items():
            print("desc: %s"%desc)
            try:
                new_desc = self.freesound_desc_conv[desc] #TODO: make options for different api's (or reimplent in derived class)
            except:
                print("MIR key is missing in the conversion dict (freesound API translation)") #FIXME
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
        if file_chosen == "Error":
            self.errorfile()
            return 

        print( os.path.getsize(file_chosen) )
        # time.sleep(3) #wait for ffmpeg conversion . FIXME: wait process completion in get_one_by_mir() method
        
        if os.path.exists( file_chosen ) and os.path.getsize(file_chosen)>1000:
            metadata = file_chosen+" by "+ author + " - id: "+str(sound_id);
            self.logging.debug(metadata+"\n")
            self.playfile(file_chosen, metadata)
    #()
#class