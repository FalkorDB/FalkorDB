FROM debian:unstable

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential autotools-dev automake libtool pkg-config
RUN apt-get install -y check valgrind
RUN apt-get install -y peg
RUN apt-get install -y libssl-dev
RUN apt-get install -y libedit-dev
RUN apt-get install -y xsltproc

RUN useradd -m build
USER build
WORKDIR /home/build
