GraphBLAS/GraphBLAS/Config:

The `GB_config.h` file in this folder is automatically created when the
GraphBLAS `libgraphblas_matlab.so` is compiled by cmake.  It allows cmake to
configure GraphBLAS JIT so that the JIT uses the same compiler and compiler
flags to compile the JIT kernels at run time.

Do not edit the `GB_config.h` file.  See `GraphBLAS/Config/GB_config.h.in`
instead, for the source file that is used to create `GB_config.h`.

