# REPOSITORY    SIZE
# apicultor     881.2 (FIXME)

# Build cmd: docker build  api.cultor -f ./Dockerfile .


# docker with apicultor webservice as entrypoint (default)
# docker run -p 5000:5000 --name apicultor  -it --net="host" apicultor

# docker with bash entrypoint
# docker run -p 5000:5000 --name apicultor  -it --net="host" --entrypoint /bin/bash apicultor

# link samples dir
# -v ./samples:/srv/apicultor/samples

# Bash entrypoint + (--rm) to automatically remove the container when it exits.
# docker run -p 5000:5000 --name apicultor  -it --rm --net="host" --entrypoint /bin/bash apicultor

FROM gcr.io/google_containers/ubuntu-slim:0.6

ENV PORT 5000

WORKDIR /srv

# TODO: Solve py2 vs py3 in apicultor
RUN apt-get update && apt-get install -y \
    python \
    python3 \
    python-pip \
    python3-pip \
    git


RUN apt-get install -y \
    build-essential \
    libyaml-dev \
    libfftw3-dev \
	libavcodec-dev \
	libavformat-dev \
	libavutil-dev \
	libavresample-dev \
	python-dev \
	libsamplerate0-dev \
	libtag1-dev \
	python-numpy-dev \
	python-numpy \
	python-yaml \
	git \
	curl

RUN pip install --upgrade pip
RUN pip install flask flask-autodoc

RUN pip install scipy

WORKDIR /srv

# APICultor code
RUN git clone https://github.com/sonidosmutantes/apicultor

# Essentia
RUN pip install essentia
# Essentia build
#RUN git clone https://github.com/MTG/essentia
#WORKDIR /srv/essentia
#RUN python waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp
#RUN python waf install


# (optional) ssh server
#RUN apt-get install -y openssh-server

EXPOSE 5000
WORKDIR /srv/apicultor
#ENTRYPOINT python examples/MockRedPanalAPI_service.py
