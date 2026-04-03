Notes added by Christoph Grüninger, Feb 22, 2025

cpu_features added to GraphBLAS, without any changes in version 0.9.0

--------------------------------------------------------------------------------

Notes added by Tim Davis, Jan 6, 2021

Added cpu_features 0.6.0, with changes suggested by
https://github.com/google/cpu_features/pull/211 .
This version has been replaced by cpu_features 0.9.0 (see above).

For both versions:

GraphBLAS does not use the cpu_features/CMakeLists.txt to build a separate
library for cpu_features.  Instead, it #include's all the source files and
include files from cpu_features into these files:

    ../Source/cpu/GB_cpu_features.h
    ../Source/cpu/GB_cpu_features_impl.c
    ../Source/cpu/GB_cpu_features_support.c

the cpu_features code is embedded in libgraphblas.so and libgraphblas.a
directly
