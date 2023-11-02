//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_config.h: JIT configuration for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GraphBLAS/Config/GB_config.h file is configured by cmake from
// GraphBLAS/Config/GB_config.h.in.

#ifndef GB_CONFIG_H
#define GB_CONFIG_H

// GB_C_COMPILER: the C compiler used to compile GraphBLAS:
#ifndef GB_C_COMPILER
#define GB_C_COMPILER   "/usr/bin/cc"
#endif

// GB_C_FLAGS: the C compiler flags used to compile GraphBLAS.  Used
// for compiling and linking:
#ifndef GB_C_FLAGS
#define GB_C_FLAGS      " -DGBMATLAB=1  -Wundef  -std=c11 -lm -Wno-pragmas  -fexcess-precision=fast  -fcx-limited-range  -fno-math-errno  -fwrapv  -O3 -DNDEBUG -fopenmp  -fPIC "
#endif

// GB_C_LINK_FLAGS: the flags passed to the C compiler for the link phase:
#ifndef GB_C_LINK_FLAGS
#define GB_C_LINK_FLAGS " -shared "
#endif

// GB_LIB_PREFIX: library prefix (lib for Linux/Unix/Mac, empty for Windows):
#ifndef GB_LIB_PREFIX
#define GB_LIB_PREFIX   "lib"
#endif

// GB_LIB_SUFFIX: library suffix (.so for Linux/Unix, .dylib for Mac, etc):
#ifndef GB_LIB_SUFFIX
#define GB_LIB_SUFFIX   ".so"
#endif

// GB_OBJ_SUFFIX: object suffix (.o for Linux/Unix/Mac, .obj for Windows):
#ifndef GB_OBJ_SUFFIX
#define GB_OBJ_SUFFIX   ".o"
#endif

// GB_OMP_INC: -I includes for OpenMP, if in use by GraphBLAS:
#ifndef GB_OMP_INC
#define GB_OMP_INC      ""
#endif

// GB_OMP_INC_DIRS: include directories OpenMP, if in use by GraphBLAS,
// for cmake:
#ifndef GB_OMP_INC_DIRS
#define GB_OMP_INC_DIRS ""
#endif

// GB_C_LIBRARIES: libraries to link with when using direct compile/link:
#ifndef GB_C_LIBRARIES
#define GB_C_LIBRARIES  " -lm -ldl /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.so"
#endif

// GB_CMAKE_LIBRARIES: libraries to link with when using cmake
#ifndef GB_CMAKE_LIBRARIES
#define GB_CMAKE_LIBRARIES  "m;dl;/usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so;/usr/lib/x86_64-linux-gnu/libpthread.so"
#endif

#endif

