# Dependencies

## OSC Client
    $ git clone https://github.com/ptone/pyosc.git
    $ cd pyosc && sudo ./setup.py install

## liblo: Lightweight OSC implementation
    * [liblo](http://liblo.sourceforge.net/)
    * [pyliblo](http://das.nasophon.de/pyliblo/)

        $ apt-get install -y liblo-dev
        $ pip2 install cython 
        $ pip2 install pyliblo 

        $ git clone https://github.com/MTG/freesound-python
        $ cd freesound-python
        $ sudo python setup.py install

## Freesound API module
```
$ git clone https://github.com/MTG/freesound-python
$ cd freesound-python
$ sudo python setup.py install
```

## Pre-processing scripts (saves realtime processing)

    $ sudo apt-get install ffmpeg

In Raspberry Pi and other debian based systems replace by:
    $ sudo apt-get install libav-tools
    alias ffmpeg=avconv

    $ pip install ffmpeg-normalize


# Old (check)
# [RtMidi](https://pypi.python.org/pypi/python-rtmidi/)

## Linux
    $ sudo apt install librtmidi-dev # y dependencias

    $ pip install python-rtmidi
    
    $ wget https://pypi.python.org/packages/49/25/1a8b1290b51fb0d4a499c3285b635c005e30b8ff423fb116db61f3d80ca5/python-rtmidi-1.1.0.zip#md5=dac7edb268a8dcd454fbeeb19ac6fb07
    $ unzip python-rtmidi-1.1.0.zip && cd python-rtmidi-1.1.0
    $ python setup.py install

## Mac
    $ brew install rtmidi # lib en c
    $ pip2 install python-rtmidi
    # Nota: rtmidi bindings para py versión python-rtmidi-0.5b1. La 1.0.0 puede tener problema con threads
    $ wget https://pypi.python.org/packages/6f/39/f7f52c432d4dd95d27703608af11818d99db0b2163cec88958efcf7c10cf/python-rtmidi-0.5b1.zip#md5=dba5808d78c843254455efb147fe87b2
    $ unzip python-rtmidi-0.5b1.zip && cd python-rtmidi-0.5b1
    $ python setup.py install

## Windows
    # python rtmidi version 1.0.0rc1
    # Lo siguiente no funciona, tampoco la versión 0.5b
    $ pip install python_rtmidi 

### Hay que compilar la librería
    $ wget https://pypi.python.org/packages/70/00/4245aedfa5d352cdb086b3a7f329e0446bd13995d2ef69fe3c2a46ca6cee/python-rtmidi-1.0.0rc1.zip#md5=f490ee1a6f8b8e83da3632fe42a203c3
    $ unzip python-rtmidi-1.0.0rc1.zip

Compilar rtmidi con Visual Studio 2015 Community Edition no funciona. Hay que instalar el Visual C++ 2010 Express de la siguiente forma:
* Bajar e instalar http://download.microsoft.com/download/1/D/9/1D9A6C0E-FC89-43EE-9658-B9F0E3A76983/vc_web.exe
* Desinstalar todo los paquetes “Microsoft Visual C++ 2010 Redistributable”
* Instalar el SDK 7.1, pero en windows 10 el instalador web se confunde con las dependencias de .net y  framework 4, entonces hay que bajar el .iso (https://download.microsoft.com/download/F/1/0/F10113F5-B750-4969-A255-274341AC6BCE/GRMSDKX_EN_DVD.iso, chequear que sea GRMSDKX_EN_DVD.iso , la X es de 64 bits) montarlo e instalar desde ahi. NO usar el setup.exe que esta en el raiz, sino Setup\SDKSetup.exe
* Ir al inicio y abrir una consola “Windows SDK 7.1 Command Prompt” (C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Microsoft Windows SDK v7.1\Windows SDK 7.1 Command Prompt) Que ejecuta: C:\Windows\System32\cmd.exe /E:ON /V:ON /T:0E /K "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd"
Nota: Sobre este problema en la web hay mucho escrito, pero para más referencia sobre como instalar dependencias con python 3.4 en Windows10, esto creo que es lo mejorcito:
https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/
y
http://haypo-notes.readthedocs.io/python.html#build-a-python-wheel-package-on-windows


* Desde una consola “Windows SDK 7.1 Command Prompt” hacer:
    $ cd python-rtmidi-1.0.0rc1.zip
    $ python setup.py install

# ffmpeg (wav conversion)

    $ sudo apt-get install ffmpeg

# (optional) Add custom MIDI external controller defs

Add custom classes to manage MIDI devices

    $ sudo cp Extensions/* /usr/local/share/SuperCollider/Extensions 

# (optional) [Pyo](http://ajaxsoundstudio.com/software/pyo/): dedicated Python module for digital signal processing

      $ sudo apt-get install python-dev libjack-jackd2-dev libportmidi-dev portaudio19-dev liblo-dev libsndfile-dev python-dev python-tk python-imaging-tk python-wxgtk2.8

      $ wget http://ajaxsoundstudio.com/downloads/pyo_0.8.5-src.tar.bz2

      $ tar -xvf pyo_0.8.5-src.tar.bz2 && cd pyo_0.8.5-src

      #enable-jack compilation
      $ sudo python setup.py --use-jack install

