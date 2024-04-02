//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_config.h: JIT configuration for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GraphBLAS/Config/GB_config.h file is configured by cmake from
// GraphBLAS/Config/GB_config.h.in.

#ifndef GB_CONFIG_H
#define GB_CONFIG_H

// GB_C_COMPILER: the C compiler used to compile GraphBLAS:
#ifndef GB_C_COMPILER
#define GB_C_COMPILER   "/opt/homebrew/opt/llvm/bin/clang"
#endif

// GB_C_FLAGS: the C compiler flags used to compile GraphBLAS.  Used
// for compiling and linking:
#ifndef GB_C_FLAGS
#define GB_C_FLAGS      " -Wno-pointer-sign  -O3 -DNDEBUG -Wno-extra-semi-stmt -fPIC  -arch arm64  -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX14.4.sdk  -fopenmp=libomp"
#endif

// GB_C_LINK_FLAGS: the flags passed to the C compiler for the link phase:
#ifndef GB_C_LINK_FLAGS
#define GB_C_LINK_FLAGS "-L/opt/homebrew/opt/llvm/lib -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -L/opt/homebrew/opt/zlib/lib -L/opt/homebrew/opt/openssl@3/lib -L/opt/homebrew/opt/readline/lib -L/opt/homebrew/opt/libiconv/lib -L/opt/homebrew/opt/gettext/lib -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -L/opt/homebrew/opt/zlib/lib -L/opt/homebrew/opt/openssl@3/lib -L/opt/homebrew/opt/readline/lib -L/opt/homebrew/opt/libiconv/lib -L/opt/homebrew/opt/gettext/lib -dynamiclib "
#endif

// GB_LIB_PREFIX: library prefix (lib for Linux/Unix/Mac, empty for Windows):
#ifndef GB_LIB_PREFIX
#define GB_LIB_PREFIX   "lib"
#endif

// GB_LIB_SUFFIX: library suffix (.so for Linux/Unix, .dylib for Mac, etc):
#ifndef GB_LIB_SUFFIX
#define GB_LIB_SUFFIX   ".dylib"
#endif

// GB_OBJ_SUFFIX: object suffix (.o for Linux/Unix/Mac/MinGW, .obj for MSVC):
#ifndef GB_OBJ_SUFFIX
#define GB_OBJ_SUFFIX   ".o"
#endif

// GB_OMP_INC: -I includes for OpenMP, if in use by GraphBLAS:
#ifndef GB_OMP_INC
#define GB_OMP_INC      ""
#endif

// GB_OMP_INC_DIRS: include directories for OpenMP, if in use by GraphBLAS,
// for cmake:
#ifndef GB_OMP_INC_DIRS
#define GB_OMP_INC_DIRS ""
#endif

// GB_C_LIBRARIES: libraries to link with when using direct compile/link:
#ifndef GB_C_LIBRARIES
#define GB_C_LIBRARIES  " -ldl /opt/homebrew/opt/llvm/lib/libomp.dylib"
#endif

// GB_CMAKE_LIBRARIES: libraries to link with when using cmake
#ifndef GB_CMAKE_LIBRARIES
#define GB_CMAKE_LIBRARIES  "dl;/opt/homebrew/opt/llvm/lib/libomp.dylib"
#endif

// GB_CUDA_COMPILER: the CUDA compiler to compile CUDA JIT kernels:
#ifndef GB_CUDA_COMPILER
#define GB_CUDA_COMPILER ""
#endif

// GB_CUDA_FLAGS: the CUDA flags to compile CUDA JIT kernels:
#ifndef GB_CUDA_FLAGS
#define GB_CUDA_FLAGS ""
#endif

// GB_CUDA_INC: -I includes for CUDA JIT kernels:
#ifndef GB_CUDA_INC
#define GB_CUDA_INC ""
#endif

// GB_CUDA_ARCHITECTURES: the CUDA ARCHITECTURES for CUDA JIT kernels:
#ifndef GB_CUDA_ARCHITECTURES
#define GB_CUDA_ARCHITECTURES ""
#endif

#endif

