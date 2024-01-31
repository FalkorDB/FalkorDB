FROM --platform=$TARGETPLATFORM falkordb/falkordb as builder

ENV DEPS "automake peg libtool autoconf python python-setuptools python-pip python-dev wget build-essential cmake m4 git valgrind"

# Set up a build environment
RUN set -ex ;\
    deps="$DEPS ";\
    apt-get update -qq ;\
    apt-get install -y --no-install-recommends $deps ;\
    pip install wheel ;\
    pip install Click rmtest backports.csv psutil behave ;\
    pip install redis-py-cluster ;\
    pip install git+https://github.com/RedisGraph/redisgraph-py.git@master ;\
    pip install git+https://github.com/RedisLabsModules/RLTest.git@master

COPY tests/requirements.txt /tmp/requirements.txt

RUN set -ex ;\
    pip install -r /tmp/requirements.txt

RUN set -ex ;\
    make clean ALL=1 ;\
    make -j `nproc`
