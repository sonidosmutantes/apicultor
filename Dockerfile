# docker build -t apicultor_v0.9 .
# REPOSITORY    SIZE
# apicultor     881.2 MB
# docker run -p 5000:5000 --name apicultor  -it --net="host"  apicultor_v0.9

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

RUN git clone https://github.com/sonidosmutantes/apicultor

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

RUN pip2 install --upgrade pip
RUN pip2 install flask flask-autodoc

# (optional) ssh server
RUN apt-get install -y openssh-server

EXPOSE 5000
ENTRYPOINT cd apicultor && python MockRedPanalAPI_service.py
