FROM debian:unstable

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential autotools-dev automake libtool pkg-config
RUN apt-get install -y check valgrind
RUN apt-get install -y libssl-dev
RUN apt-get install -y libedit-dev
RUN apt-get install -y git-buildpackage
RUN apt-get install -y vim
RUN apt-get install -y doxygen

RUN useradd -m build
USER build
RUN git config --global user.name "Chris Leishman"
RUN git config --global user.email "chris@leishman.org"
ENV DEBFULLNAME "Chris Leishman"
ENV DEBEMAIL "chris@leishman.org"
WORKDIR /home/build
