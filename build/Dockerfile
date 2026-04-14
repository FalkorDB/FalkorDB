FROM redis:8.6.2

ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt update -y && \
    apt install -y lsb-release gnupg wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 22 && \
    apt update -y

RUN apt install -y git make cmake curl zlib1g zlib1g-dev libpolly-22-dev libomp-22-dev libc++-22-dev libc++abi-22-dev libzstd-dev python3-venv lcov

RUN python3 -m venv venv

RUN venv/bin/pip install behave falkordb falkordb-bulk-loader hypothesis pytest pytest-benchmark RLTest

RUN git clone --branch dev2 --single-branch https://github.com/DrTimothyAldenDavis/GraphBLAS.git
# RUN git clone --branch v10.3.1 --single-branch https://github.com/DrTimothyAldenDavis/GraphBLAS.git

RUN cd GraphBLAS && \
    make static CMAKE_OPTIONS='-DGRAPHBLAS_COMPACT=1 -DCMAKE_POSITION_INDEPENDENT_CODE=on' JOBS=2 CC=clang-22 CXX=clang++-22 && \
    make install && \
    cd .. && \
    rm -rf GraphBLAS

RUN git clone --branch v1.2.1 --single-branch https://github.com/GraphBLAS/LAGraph.git && \
    cd LAGraph && \
    mkdir -p build && \
    cd build && \
    cmake .. \
        -DBUILD_STATIC_LIBS=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DLIBRARY_ONLY=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGRAPHBLAS_INCLUDE_DIR=/usr/local/include/suitesparse \
        -DGRAPHBLAS_LIBRARY=/usr/local/lib/libgraphblas.a \
        -DSUITESPARSE_USE_FORTRAN=OFF \
        -DCMAKE_C_COMPILER=clang-22 \
        -DCMAKE_CXX_COMPILER=clang++-22 && \
    make -j2 && \
    mkdir -p /data/lagraph_lib && \
    cp src/liblagraph.a /data/lagraph_lib/ && \
    cp experimental/liblagraphx.a /data/lagraph_lib/ && \
    cp ../include/LAGraph.h /data/lagraph_lib/ && \
    cp ../include/LAGraphX.h /data/lagraph_lib/ && \
    cd /data && \
    rm -rf LAGraph

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN . /data/venv/bin/activate && \
    mkdir -p redisearch && \
    cd redisearch && \
    git clone --recurse-submodules --branch vector-low-level-api --single-branch https://github.com/FalkorDB/RediSearch.git && \
    cd RediSearch && \
    sed -i 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt && \
    make build STATIC=1 CLANG=1 CC=clang-22 CXX=clang++-22

RUN cd redisearch && \
    cp -r RediSearch/bin bin  && \
    rm -rf RediSearch

CMD "/bin/bash"
