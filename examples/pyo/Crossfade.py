"""
Crossfade
"""
from pyo import *

s = Server().boot()

#s.start()
#sf = SfPlayer(SNDS_PATH + "/transparent.aif", loop=True, mul=.4).out()
#fol = Follower(sf, freq=30)
#n = Noise(mul=fol).out(1)

#sf = SfPlayer(SNDS_PATH + "/transparent.aif", loop=True, mul=.4).out()
#fol2 = Follower2(sf, risetime=0.002, falltime=.1, mul=.5)
#n = Noise(fol2).out(1)

a = SfPlayer("File1.wav", loop=True, mul=1).out()
b = SfPlayer("File2.wav", loop=True, mul=1).out()

c = Clip(a).out()

c.setInput(b, fadetime=3) # 3 seconds crossfade

s.gui(locals())
