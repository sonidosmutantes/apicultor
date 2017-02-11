# Requirements

Tested under Debian, Ubuntu 15.04 and 16.04 but should work in other operating systems

# PIP 
$ sudo apt-get install python3-pip python-pip

### (optional) create a virtualenv
$ virtualenv apicultor_venv
$ source apicultor_venv/bin/activate

### Dependencias via pip
$ pip install -i requirements.txt

## Mock RedPanal WebService 

$ sudo pip2 install flask

### Doc

$ sudo pip2 install flask-autodoc

## MIR

### Essentia (http://essentia.upf.edu/)

$ git clone https://github.com/MTG/essentia.git

You can install those dependencies on a Debian/Ubuntu system from official repositories using the commands provided below:
$ sudo apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev

In order to use python bindings for the library, you might also need to install python-numpy-dev (or python-numpy on Ubuntu) and python-yaml for YAML support in python:
$ sudo apt-get install python-numpy-dev python-numpy python-yaml

./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp 

To compile everything you’ve configured:
$ ./waf

To install the C++ library and the python bindings (if configured successfully; you might need to run this command with sudo):
$ sudo ./waf install

# Database
$ sudo apt-get install python-mysqldb

## Crear una base de datos:

Si es tu primera vez utilizando bases de datos de MySQL, tenés que instalar MySQL en tu sistema y luego crear un usuario con una password para acceder y después crear la base de datos a la que accederás utilizando el usuario y la password.

Luego de correr sudo apt-get install mysql-server:
```
$ mysql // a veces el comando puede ser mysqld de acuerdo al paquete instalado
mysql> CREATE USER 'usuario'@'localhost' IDENTIFIED BY 'password';
mysql> CREATE DATABASE nombredelabasededatos;
mysql> GRANT ALL PRIVILEGES ON nombredelabasededatos.* TO 'usuario'@'localhost';
mysql> quit;
```

Luego se puede usar `Fill_DB.py` para crear la base de datos del MIR


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

## State Machine

$ sudo pip install git+git://github.com/riccardoscalco/Pykov@master #both Python2 and Python3

Note that Pykov depends on numpy and scipy

## OSC

# OSC Service Example
$ sudo pip3 install python-osc

# State Machine Example
$ git clone https://github.com/ptone/pyosc.git
$ cd pyosc && sudo ./setup.py install

## Supercollider

$ sudo apt-get install supercollider supercollider-ide
$ sudo apt-get install scide 

## Pre-processing

$ sudo apt-get install ffmpeg

