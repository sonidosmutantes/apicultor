"""
Automatic Gain Control

"""
from pyo import *


class AGC(PyoObject):
    """
    AGC.

  

    """


    def __init__(self, input, int=1., mul=1, add=0):
        # Properly initialize PyoObject's basic attributes
        PyoObject.__init__(self)

        # Keep references of all raw arguments
        self._input = input
        self._int = int

        # Using InputFader to manage input sound allows cross-fade when changing sources
        self._in_fader = InputFader(input)

        # Convert all arguments to lists for "multi-channel expansion"
        in_fader, int, mul, add, lmax = convertArgsToLists(self._in_fader, int, mul, add)

        #spoint= 1
        #gain = 0.001 #starter gain
        spoint= 0.25
        gain = 0.1 #starter gain
        
        # Apply processing
        self._y = self._input * self._int 
        err = spoint - Abs(self._y)
        j = self._int 
        self._int = j+gain*err
        
#        self._y = in_fader * int 
#        err = spoint - Abs(self._y)
#        j = int 
#        int = j+gain*err

#        #alternate implementation
#        self._y = self._input * self._int 
#        #err = spoint - Abs(self._y)
#        err = spoint - self._y*self._y
#        j = self._int
#        self._int = j + gain*err 

        self._agc = Interp(in_fader, self._y, mul=mul, add=add)

        
        # self._base_objs is the audio output seen by the outside world!
        self._base_objs = self._agc.getBaseObjects()


    def setInput(self, x, fadetime=0.05):
        """

        """
        self._input = x
        self._in_fader.setInput(x, fadetime)
    
    def setInt(self, i):
        """
        Replace the `integrate` attribute.

        :Args:

            x : float or PyoObject
                New `int` attribute.

        """
        self._int = i
        

    @property
    def input(self): 
        """PyoObject. Input signal to process."""
        return self._input
    @input.setter
    def input(self, x): 
        self.setInput(x)

    @property
    def int(self): 
        """float or PyoObject. Int."""
        return self._int
    @int.setter
    def int(self, x): 
        self.setInt(x)


    def ctrl(self, map_list=None, title=None, wxnoserver=False):
        self._map_list = [SLMap(0., 1., "lin", "depth", self._int),
                          SLMapMul(self._mul)]
        PyoObject.ctrl(self, map_list, title, wxnoserver)

    def play(self, dur=0, delay=0):
        self._y.play(dur, delay)
        return PyoObject.play(self, dur, delay)

    def stop(self):
        self._y.stop()
        return PyoObject.stop(self)

    def out(self, chnl=0, inc=1, dur=0, delay=0):
        self._y.play(dur, delay)
        return PyoObject.out(self, chnl, inc, dur, delay)

# Run the script to test the AGC object.
if __name__ == "__main__":
    s = Server().boot()
    #src = BrownNoise([.2,.2]).out()
    samples_path = "/Users/hordia/git/apicultor/samples/"
    file_path = samples_path + "256_sample3.wav"
    t = SndTable(file_path)
    sr = t.getRate()
    c = TableRead(table=t, freq=[sr,sr*.99], loop=True, mul=1)
    #c.out()

    fl = AGC(c, int=1, mul=1).out()
    
    s.start()
    s.gui(locals())