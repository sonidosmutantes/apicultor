# Docker build (ETA 3hs)

Tutorial para "dockerizar" apiCultor by [azimut].

## Motivación

Si uno no tiene GNU+Linux o en particular Ubuntu, se puede "virtualizar" con [docker] bastante facilmente.

## Dependencias

Se asume que se tienen clonados los repos de Essentia y apiCultor en los siguientes paths:
```
$ cd
$ mkdir -p projects
$ git clone https://github.com/MTG/essentia $HOME/projects/essentia
$ git clone https://github.com/sonidosmutantes/apicultor $HOME/projects/apicultor
```

## Docker build
Se crea un nuevo container
```
$ sudo docker run  --name essentia -v $HOME/projects/essentia/:/opt/essentia -v $HOME/projects/apicultor/:/opt/apicultor -ti gcr.io/google_containers/ubuntu-slim:0.6 /bin/bash
```
### Essentia build
Una vez dentro se siguen los pasos para instalar las dependencias y buildear [essentia].
```
# apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev
# apt-get install python-numpy-dev python-numpy python-yaml
# cd /opt/essentia
# ./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp
# ./waf install
```
### apiCultor
(En docker)
```
# apt-get install python-pip ffmpeg
# pip install bs4 regex wget matplotlib numpy scipy scikit-learn colorama librosa transitions
# pip install smst
```

## Notas sobre mantenimiento y TODOs

* Agregar más pip o apt-get que falten
* Se usa [ubuntu-slim] porque se prefiere una imágen simple como base, probablemente haya que migrar a la oficial de ubuntu
* No se usa un `Dockerfile` principalmente porque `essentia`(600MB) y `apicultor`(130MB) son un tanto pesados para bajar y encima tener que poner en una imagen de Docker. Si algun dia `docker build` soporta volumenes o alguna de las otras opciones es usable habría que usar un `Dockerfile`.
  * Ver opcin de descargar el master .zip y usar un `Dockerfile`...
  * Si se migrase a un Dockerfile, probablemente habría que usar uno para essentia y uno para apicultor que use el anterior como base
* Pushear la imagen a dockerhub.
* Medir el tiempo mejor
* Essentia es compilado staticamente (segun el flag) y los binding pueden moverse a un egg lo que haría posible distribuir esto mas fácil
  * (ver si no esta hecho ya)

[azimut]: https://github.com/azimut
[docker]: https://docs.docker.com/engine/installation/
[essentia]: http://essentia.upf.edu/documentation/installing.html
[ubuntu-slim]: https://github.com/kubernetes/contrib/blob/master/images/ubuntu-slim/Dockerfile.build
