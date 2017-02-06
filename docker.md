
   * [Docker build (ETA 3hs)](#docker-build-eta-3hs)
      * [Preparar repos](#preparar-repos)
      * [Essentia](#essentia)
         * [build](#build)
      * [apicultor](#apicultor)
      * [Notas de mantenimiento y TODOs](#notas-de-mantenimiento-y-todos)

# Docker build (ETA 3hs)

Si uno no tiene Ubuntu, se puede "virtualizar" ubuntu con [docker] bastante facilmente.

## Preparar repos
Se asume que:
```
$ cd
$ mkdir -p projects
$ git clone https://github.com/MTG/essentia projects/essentia
$ git clone https://github.com/sonidosmutantes/apicultor projects/apicultor
```

## Essentia
### build
Creo un nuevo container y una vez dentro sigo los steps para instalar las dependencias para buildear [essentia].
```
$ sudo docker run  --name essentia -v $HOME/projects/essentia/:/opt/essentia -v $HOME/projects/apicultor/:/opt/apicultor -ti gcr.io/google_containers/ubuntu-slim:0.6 /bin/bash
# apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev
# apt-get install python-numpy-dev python-numpy python-yaml
# cd /opt/essentia
# ./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp
# ./waf install
```
## apicultor
En docker
```
# apt-get install python-pip ffmpeg
# pip install bs4 regex wget matplotlib numpy scipy scikit-learn colorama librosa transitions
# pip install smst
```

## Notas de mantenimiento y TODOs

* Agregar mas pip o apt-get que falten
* Enfocarse en cosas no graficas por ahora
* Uso [ubuntu-slim] porque prefiero una imagen simple como base, probablemente haya que migrar a la oficial de ubuntu
* No uso un `Dockerfile` principalmente porque es `essentia`(600MB) y `apicultor`(130MB) son un tanto pesados para bajar y encima tener que poner en una imagen de Docker. Si algun dia `docker build` soporta volumenes o alguna de las otras opciones es usable habria que usar un `Dockerfile`.
* Tabla creada con [github-markdown-toc]
* Alguien tiene que pushear la imagen a dockerhub.
* Medir el tiempo mejor
* Podria descargar el master .zip y usar un `Dockerfile`...
* Si se migrase a un Dockerfile, problablemente habria que usar uno para essentia y uno para apicultor que use el aterior como base
* essentia es compilado staticamente (segun el flag) y los binding pueden moverse a un egg lo que haria posible distribuir esto mas facil...pero es trabajo de esentia

[docker]: https://docs.docker.com/engine/installation/
[github-markdown-toc]: https://github.com/ekalinin/github-markdown-toc
[essentia]: http://essentia.upf.edu/documentation/installing.html
[ubuntu-slim]: https://github.com/kubernetes/contrib/blob/master/images/ubuntu-slim/Dockerfile.build
