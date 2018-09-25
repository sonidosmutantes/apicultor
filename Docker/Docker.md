
## Docker

See [docker](docker.md) and [Dockerfile](Dockerfile.md).


API listening in port 5000:
```
$ docker build -t apicultor_v0.9 .
$ docker run -p 5000:5000 --name apicultor  -it --net="host"  apicultor_v0.9
```
