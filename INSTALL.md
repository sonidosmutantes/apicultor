# Genearl requirements

Python 3.6
Note: old versión is Python 2.7 but is deprecated

### Operating System
Tested under Linux, Mac OS (>10.11) and Windows 10.

Debian, Ubuntu 15.04 and 16.04 (and .10). And [Docker](docker.md) images.
Raspian @ Raspberry Pi.

# Install

Optionally, create a virtualenv with python3 as binary

    $ virtualenv -p python3 [SOME_PATH]/ApiCultor_dev
    $ source [SOME_PATH]/ApiCultor_dev/bin/activate

Install the apicultor module:

    $ python setup.py install


# Dependencies requeriments

## Linux

    $ sudo apt-get install python3-pip python-pip

## Mock RedPanal WebService 

    $ sudo pip2 install flask

### Doc

    $ sudo pip2 install flask-autodoc

# (optional) Pyo

If you want to run Pyo examples

    $ sudo apt-get install python-dev libjack-jackd2-dev libportmidi-dev portaudio19-dev liblo-dev libsndfile-dev python-dev python-tk python-imaging-tk python-wxgtk2.8

    $ wget http://ajaxsoundstudio.com/downloads/pyo_0.8.5-src.tar.bz2

    $ tar -xvf pyo_0.8.5-src.tar.bz2 && cd pyo_0.8.5-src

#enable-jack compilation

    $ sudo python setup.py --use-jack install

# MIR

### Essentia (http://essentia.upf.edu/)

	$ pip install essentia

#### MAC

	brew tap MTG/essentia
	brew install essentia

Reference: https://github.com/MTG/homebrew-essentia

#### Linux build

    $ git clone https://github.com/MTG/essentia.git

You can install those dependencies on a Debian/Ubuntu system from official repositories using the commands provided below:

    $ sudo apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev

In order to use python bindings for the library, you might also need to install python-numpy-dev (or python-numpy on Ubuntu) and python-yaml for YAML support in python:

    $ sudo apt-get install python-numpy-dev python-numpy python-yaml

    $ ./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp 

To compile everything you’ve configured:

    $ ./waf

To install the C++ library and the python bindings (if configured successfully; you might need to run this command with sudo):

    $ sudo ./waf install

# Database

    $ sudo apt-get install python-mysqldb

## Crear una base de datos:

Si es tu primera vez utilizando bases de datos de MySQL, tenés que instalar MySQL en tu sistema y luego crear un usuario con una password para acceder y después crear la base de datos a la que accederás utilizando el usuario y la password.

Luego de correr sudo apt-get install mysql-server:

    $ mysql // a veces el comando puede ser mysqld de acuerdo al paquete instalado
    mysql> CREATE USER 'usuario'@'localhost' IDENTIFIED BY 'password';
    mysql> CREATE DATABASE nombredelabasededatos;
    mysql> GRANT ALL PRIVILEGES ON nombredelabasededatos.* TO 'usuario'@'localhost';
    mysql> quit;

Luego se puede usar `Fill_DB.py` para crear la base de datos del MIR

# (old-deprectaed) Without installing the module

### Dependencias viaa pip

    $ pip install -i requirements.txt


### Analysis

* pip install bs4
* pip install regex
* pip install wget
* pip install matplotlib
* pip install numpy scipy scikit-learn
* pip install smst
* pip install colorama

    $ sudo pip install bs4 regex wget numpy scipy scikit-learn 
    $ sudo pip install matplotlib smst
    $ sudo apt-get install python-tk

* pip install librosa
* pip install transitions

* Essentia (ver instrucciones para compilar aquí: http://essentia.upf.edu/documentation/installing.html)

## MIR State Machine example

### pykov (markov processes)
Note that Pykov depends on numpy and scipy.

Both for Python2 and Python3:

    $ sudo pip install git+git://github.com/riccardoscalco/Pykov@master 

Note: In Raspberry Pi run first $ sudo apt-get install python-numpy python-scipy #pip install scipy no works

### liblo: Lightweight OSC implementation

    $ apt-get install -y liblo-dev
  
    $ pip2 install cython 
    $ pip2 install pyliblo 

#### Freesound API module

    $ git clone https://github.com/MTG/freesound-python
    $ cd freesound-python
    $ sudo python setup.py install

# OSC (different libraries)

## OSC Service Example

    $ sudo pip3 install python-osc

## OSC Client

    $ git clone https://github.com/ptone/pyosc.git
    $ cd pyosc && sudo ./setup.py install

## Supercollider

    $ sudo apt-get install supercollider supercollider-ide
    $ sudo apt-get install scide 

## Pre-processing scripts (saves realtime processing)

    $ sudo apt-get install ffmpeg
    $ pip2 install ffmpeg-normalize

In Raspberry Pi, Bela and other debian based systems replace by:
    $ sudo apt-get install libav-tools
    
Add in ~/.bashrc:

    alias ffmpeg=avconv

