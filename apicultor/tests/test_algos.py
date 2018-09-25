from ..utils.algorithms import *
from ..sonification.Sonification import write_file
from soundfile import read

audio, fs = read(sys.argv[1])
audio = mono_stereo(audio)

song = sonify(audio, fs)

def test_Mel_Filter():   
    song.mel_bands_global()                   
    for i in song.filter_coef:                    
        if (len(np.nonzero(i)[0]) != 0) == False:
            return False
    return 'OK'

def test_FFT():
    song.frame = audio[:1024]
    song.window()
    result = np.std(fft(song.windowed_x, 2048) - son.fft(song.windowed_x)) < 1e-04 #if the FFT is awesome, most of processing will be
    return result

class test_algos(unittest.TestCase):
    def test_everything(self):
        self.assertEqual(test_Mel_Filter == 'OK',True) #assert there are ascending numbers of filter coefficients
        self.assertEqual(test_FFT == False,False) #assert FFT result is similar to numpy's fft.fft

Usage = "python3 test_algos.py SoundFilename.ogg/wav" 
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nCan't test without Music!\n", Usage, "\n")
        sys.exit(1)
    unittest.main()
