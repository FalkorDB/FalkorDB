Notes added by Tim Davis, Jan 6, 2021

cpu_features added to GraphBLAS, with changes suggested by
https://github.com/google/cpu_features/pull/211 , which renames
./src/impl_aarch64_linux_or_android.c to ./src/impl_aarch64.c renamed; and
modifies it:

18c18
< #if defined(CPU_FEATURES_OS_LINUX) || defined(CPU_FEATURES_OS_ANDROID)
---
> #if defined(CPU_FEATURES_OS_LINUX) || defined(CPU_FEATURES_OS_ANDROID) || defined(CPU_FEATURES_OS_MACOS)
149c149
< #endif  // defined(CPU_FEATURES_OS_LINUX) || defined(CPU_FEATURES_OS_ANDROID)
---
> #endif  // defined(CPU_FEATURES_OS_LINUX) || defined(CPU_FEATURES_OS_ANDROID) || defined(CPU_FEATURES_OS_MACOS)

cpu_features/test/CMakeLists.txt is also modified:

diff -ar ../../cpu_features/test/CMakeLists.txt ./test/CMakeLists.txt
74c74
<   add_executable(cpuinfo_aarch64_test cpuinfo_aarch64_test.cc ../src/impl_aarch64_linux_or_android.c)
---
>   add_executable(cpuinfo_aarch64_test cpuinfo_aarch64_test.cc ../src/impl_aarch64.c)

Makefile added by Tim Davis (no longer needed however)

GraphBLAS does not use the cpu_features/CMakeLists.txt to build a separate
library for cpu_features.  Instead, in #include's all the source files and
include files from cpu_features into these files:

    ../Source/cpu/GB_cpu_features.h
    ../Source/cpu/GB_cpu_features_impl.c
    ../Source/cpu/GB_cpu_features_support.c

the cpu_features code is embedded in libgraphblas.so and libgraphblas.a
directly
