class MIRState:
    debug = True
    desc = dict() # MIR Descriptors
    enabled_descriptors = [
        "duration",
        "BPM",
        "hfc.mean",
        "spectral_complexity.mean",
        "spectral_centroid.mean",
        "pitch_centroid.mean",
        "inharmonicity.mean"
    ]

    default_sign = {
        "duration": "<",
        "BPM": "<",
        "hfc.mean": "=",
        "spectral_complexity.mean": "=",
        "spectral_centroid.mean": "=",
        "pitch_centroid.mean": "=",
        "inharmonicity.mean": "="
    }

    def __init__(self):
        self.available_descriptors = list()
        self.sign = dict()
        self.enabled = dict()
        for item in self.__class__.enabled_descriptors:
           item = item.lower()
           self.available_descriptors.append(item)
           self.available_descriptors.append(item+'.enabled')
           self.available_descriptors.append(item+'.mod')
    #()

    def set_desc(self, name, value):
        if name not in self.available_descriptors:
            raise NotImplementedError

        if '.' not in name:
            self.desc[name] = value
            if name not in self.enabled: #TODO: check this inicialization
                self.enabled[name] = 1.
                self.sign[name] = self.default_sign[name]
                # self.sign[name] = "<"
            if self.debug:
                print( "Updated state ", self.desc )
        else:
            c = None
            try:
                a, b = name.split('.')
            except:
                a, b, c = name.split('.')

            if b=="mean" and c==None:
                self.desc[name] = value
                if name not in self.enabled: #TODO: check this inicialization
                    self.enabled[name] = 1.
                    self.sign[name] = self.default_sign[name]
                if self.debug:
                    print( "Updated state ", self.desc )
            elif b=="enabled" or c=="enabled":
                value = int(value)
                name = a
                if c=="enabled":
                    name = a+'.'+b
                self.enabled[name] = value
                if value==0:
                    if name in self.desc:
                        del self.desc[name]
                        print("Removing key: %s"%name)
                if self.debug:
                    print("enabled value: %i"%value)
            elif b=="mod" or c=="mod":
                name = a
                if c=="mod":
                    name = a+'.'+b
                
                sign = "="
                if value==1:
                    sign = '>'
                elif value==-1:
                    sign = '<'
                else:
                    sign = '='

                self.sign[name] = sign
                if self.debug:
                    print("modificator: %s"%sign)
    #()

    def get_descriptors_values(self):
        return self.desc
    #()

    def load_json(self, jsonpath):
        """ Desserializa """
        # open(jsonpath)
        # self.desc[] =
        pass
    #()

    def save_json(self):
        """ serializa """
        pass
        # return json
    #()

