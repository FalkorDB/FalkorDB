# GraphBLAS/JITPackage:  package GraphBLAS source for the JIT

SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

The use of this package is not required by the end user.  If you edit the
GraphBLAS source code itself or build from a source tarball, however, you must
read the following instructions.

This small stand-alone package compresses all the source files (`*.c` and
`*.h`) required by the JIT kernels into a single file: `GB_JITpackage.c`.  The
CMake build system for GraphBLAS automatically generates `GB_JITpackage.c` if
it is necessary.


## Automatic JIT compilation and cache

When GraphBLAS starts, `GrB_init` checks the user source folder to ensure
`~/.SuiteSparse/GrBx.y.z/src` exists (where x.y.z is the current GraphBLAS
version number), and that it contains the GraphBLAS source code.  It does this
with a quick test: `~/.SuiteSparse/GrB.x.y.z/src/GraphBLAS.h` must exist, and
the first line is checked to see if the version matches the GraphBLAS library
version.  If the file is not present or the version does not match, `GrB_Init`
uncompresses each file from its compressed form in `GB_JITpackage.c`, and
writes it to the user JIT source folder.

If you edit the GraphBLAS source that goes into the file `GB_JITpackage.c`, you
must delete your entire cache (simply delete the `~/.SuiteSparse/GrBx.y.z`
folder), since these are updated only if the GraphBLAS version changes.
`GrB_Init` only checks the first line of
`~/.SuiteSparse/GrB.x.y.z/src/GraphBLAS.h`.  It does not check for any changes
in the rest of the code.  If the `src` folder in the cache changes, then any
prior compiled JIT kernels are invalidated.  It is also safest to delete any
`GraphBLAS/PreJIT/*` files; these will be recompiled properly if the `src`
cache files change, but any changes in other parts of GraphBLAS (the JIT
sources itself, in `GraphBLAS/Source/*fy*c`, in particular) can cause these
kernels to change.

A future version of GraphBLAS may do a more careful check (such as a CRC
checksum), so that this check would be automatic.  This would also guard
against a corrupted user cache.


## Cross-compilation

The file `GB_JITpackage.c` is generated by a binary that is built and executed
by the CMake build system.  When cross-compiling, the same compiler toolchain
that is used for the *target* cannot be used to build that generator
executable.  Instead, a different (native) compiler toolchain needs to be used
to build that generator executable.  That native compiler toolchain must be
able to produce binaries that can execute natively on the build *host* (i.e.,
without virtualization, emulation, or similar means).

To pass CMake flags that need to be used for that native build, it is possible
to use a toolchain file.  That toolchain file could look like the following:

``` cmake
# CMake settings for 64-bit x86 Linux build host
set ( CMAKE_SYSTEM_NAME "Linux" )
set ( CMAKE_SYSTEM_PROCESSOR "x86_64" )
set ( CMAKE_C_COMPILER "gcc" )
set ( CMAKE_C_FLAGS "" CACHE STRING "" FORCE )
set ( CMAKE_EXE_LINKER_FLAGS "" CACHE STRING "" FORCE )
set ( CMAKE_MODULE_LINKER_FLAGS "" CACHE STRING "" FORCE )
set ( CMAKE_SHARED_LINKER_FLAGS "" CACHE STRING "" FORCE )

set ( CMAKE_CROSSCOMPILING OFF )
```

In that example, the toolchain file indicates that the build host is a 64-bit
x86 Linux and `gcc` should be used as the C compiler.  Depending on the build
environment, not all of these flags might need to be overridden, or you might
need to set or override more CMake flags for the native toolchain.

To pass the native toolchain file to CMake, save it to, e.g.,
`/some/path/to/native-toolchain.cmake`.  Then, configure with the flag
`-DGRAPHBLAS_CROSS_TOOLCHAIN_FLAGS_NATIVE="-DCMAKE_TOOLCHAIN_FILE=/some/path/to/native-toolchain.cmake"`.

If compilation fails, check the content of the file
`GraphBLAS/JITpackage/native/CMakeCache.txt` in the build tree.  If necessary,
change or override more flags in the native toolchain file.
