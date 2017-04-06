FROM gcr.io/google_containers/ubuntu-slim:0.6

RUN apt-get update && apt-get install -y build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev python-numpy-dev python-numpy python-yaml

RUN mkdir -p projects && git clone https://github.com/sonidosmutantes/apicultor $HOME/git/apicultor