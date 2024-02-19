ARG BASE_IMAGE=falkordb/falkordb-compiler

ARG TARGETPLATFORM=linux/amd64

FROM --platform=$TARGETPLATFORM $BASE_IMAGE as compiler

RUN curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg ;\
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list ;\
    apt-get update ;\
    apt-get install redis

ENV DEPS "automake peg libtool autoconf python-is-python3 2to3 python3.10-venv python3-pip wget build-essential cmake m4 git valgrind gcc python3-dev"

# Set up a build environment
RUN set -ex ;\
    deps="$DEPS ";\
    apt-get update -qq ;\
    apt-get install -y --no-install-recommends $deps ;

# Create venv
RUN set -ex ;\
    python3 -m venv /venv ;\
    . /venv/bin/activate ;

RUN set -ex ;\
    /venv/bin/pip3 install --upgrade pip ;\
    /venv/bin/pip3 install --upgrade setuptools

RUN set -ex ;\
    /venv/bin/pip3 install -r /FalkorDB/tests/requirements.txt