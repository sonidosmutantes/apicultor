FROM gcr.io/google_containers/ubuntu-slim:0.6

ENV http_proxy http://172.17.0.2:3128
ENV https_proxy http://172.17.0.2:3128

# RUN  echo 'Acquire::http { Proxy "http://localhost:3142"; };' >> /etc/apt/apt.conf.d/01proxy
# RUN apt-get update && apt-get install -y vim git
# # docker build -t my_ubuntu .

RUN apt-get update && apt-get install -y build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev python-numpy-dev python-numpy python-yaml

RUN mkdir -p projects && git clone https://github.com/sonidosmutantes/apicultor $HOME/git/apicultor
