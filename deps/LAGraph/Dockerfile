ARG SS_RELEASE=v3.2.0
ARG BASE_CONTAINER=graphblas/suitesparse-graphblas:${SS_RELEASE}
FROM ${BASE_CONTAINER}

RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY . /LAGraph
WORKDIR /LAGraph
RUN make clean
RUN make library
RUN make install
