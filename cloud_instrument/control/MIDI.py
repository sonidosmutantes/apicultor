import platform
import rtmidi
import time

class MIDI:
    """
        MIDI support
    """
    def __init__(self):
        if platform.system() == "Windows":
            midi_api = rtmidi.API_WINDOWS_MM
        elif platform.system() == "Darwin":
            midi_api = rtmidi.API_MACOSX_CORE
        else:
            midi_api = rtmidi.API_UNSPECIFIED

        midiin = rtmidi.MidiIn(midi_api)
    
        # midiin.set_callback(__class__.MidiInputHandler(port_name, args.config))
        midiin.set_callback(lambda msg, t: self.process(msg))

        # port = 0
        # name = "CloudInstrument"
        # midiin, port_name = open_midiinput(
        #                         port,
        #                         use_virtual=True,
        #                         api=BACKEND_MAP.get(args.backend, rtmidi.API_UNSPECIFIED),
        #                         client_name=name,
        #                         port_name='MIDI input')
    #()
    
    @classmethod
    def process(self,msg):
        print(msg)
        pass