# Dockerfile (camino rápido)

Servicio (API) escuchando en puerto 5000:
```
$ docker build -t apicultor_v0.9 .
$ docker run -p 5000:5000 --name apicultor  -it --net="host"  apicultor_v0.9
```

# Docker build. Paso a paso (ETA 30min)

Tutorial para "dockerizar" apiCultor by [azimut].

## Motivación

Si uno no tiene GNU+Linux o en particular Ubuntu, se puede "virtualizar" con [docker] bastante facilmente.

## Dependencias

Se asume que se tienen clonados los repos de Essentia y apiCultor en los siguientes paths:

```
$ cd
$ mkdir -p projects
$ git clone https://github.com/MTG/essentia $HOME/git/essentia
$ git clone https://github.com/sonidosmutantes/apicultor $HOME/git/apicultor
```

## Docker run

Se crea un nuevo container de nombre apicultor. Se hace port forwarding entre el puerto 5000 del container y del host.

```
$ sudo docker run -p 5000:5000 --name apicultor -v $HOME/git/essentia/:/opt/essentia -v $HOME/git/apicultor/:/opt/apicultor -ti gcr.io/google_containers/ubuntu-slim:0.6 /bin/bash
```

Si se usa Linux/Ubuntu como host, agregar:  
``` 
--net="host"
```

### Essentia build
Una vez dentro se siguen los pasos para instalar las dependencias y buildear [essentia].

```
# apt-get update
# apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev
# apt-get install python-numpy-dev python-numpy python-yaml
# cd /opt/essentia
# ./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp
# ./waf install
```

### ApiCultor
(En docker)

```
# apt-get install python-pip ffmpeg
# pip install bs4 regex wget matplotlib numpy scipy scikit-learn colorama librosa transitions
# pip install smst
```
 
#### (Opcional)

```
# pip2 install flask flask-autodoc
# apt-get install net-tools git
```

## Docker commit

```
Obtener ID y commit
$ docker ps
$ docker commit [ID]
```

## Lanzar una nueva terminal

```
$ docker exec -it apicultor bash
```

## Lanzar el mock ApiCultor web service (port: 5000)

```
$ docker exec -it apicultor /opt/run_ws.sh
```

Donde run_ws.sh:

```
cd /opt/apicultor
./MockRedPanalAPI_service.py
```

## Notas sobre mantenimiento y TODOs (by [azimut])
* Se usa [ubuntu-slim] porque se prefiere una imágen simple como base.
* No se usa un `Dockerfile` principalmente porque `essentia`(600MB) y `apicultor`(130MB) son un tanto pesados para bajar y encima tener que poner en una imagen de Docker. Si algún día `docker build` soporta volumenes o alguna de las otras opciones es usable, habría que usar un `Dockerfile`.
 * Ver opción de descargar el master .zip y usar un `Dockerfile`.
 * Si se migrase a un Dockerfile, probablemente habría que usar uno para essentia y uno para apicultor que use el anterior como base.
* Pushear la imagen a dockerhub.
* "Essentia es compilado estáticamente (segun el flag) y los binding pueden moverse a un egg lo que haría posible distribuir esto mas fácil" (ver si no esta hecho ya).
* Medir el tiempo mejor.

[azimut]: https://github.com/azimut
[docker]: https://docs.docker.com/engine/installation/
[essentia]: http://essentia.upf.edu/documentation/installing.html
[ubuntu-slim]: https://github.com/kubernetes/contrib/blob/master/images/ubuntu-slim/Dockerfile.build
