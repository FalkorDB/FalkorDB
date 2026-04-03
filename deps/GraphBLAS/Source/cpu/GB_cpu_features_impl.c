//------------------------------------------------------------------------------
// GB_cpu_features_impl.c: Google's cpu_features package for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Google's cpu_features package makes extensive use of bit field manipulation.
// This makes the code easy to read, but the layout of the bits can depend on
// the implementation by the compiler.  The cpu_features/CMakeLists.txt file
// has this recommendation:

    // cpu_features uses bit-fields which are to some extent implementation
    // defined (see https://en.cppreference.com/w/c/language/bit_field).  As a
    // consequence it is discouraged to use cpu_features as a shared library
    // because different compilers may interpret the code in different ways.
    // Prefer static linking from source whenever possible.
    // option(BUILD_SHARED_LIBS "Build library as shared." OFF)

// GraphBLAS avoids this issue by compiling the cpu_features source directly
// into the libgraphblas.so and libgraphblas.a compiled libraries themselves.
// This ensures that the same compiler is used for both GraphBLAS and
// cpu_features.

// This file simply #include's all of the cpu_features/src/impl_*.c files,
// one for each architecture (and multiple ones for x86).  The supporting
// files for cpu_features are #include'd by GB_cpu_features_support.c.

#include "include/GB_compiler.h"
#include "cpu/GB_cpu_features.h"
#include "include/GB_pedantic_disable.h"
#include "include/GB_unused.h"

#if !defined ( GBNCPUFEAT )

    // include the implementation files from cpu_features/src/impl_*.c
    #include "src/impl_aarch64_linux_or_android.c"
    #include "src/impl_aarch64_windows.c"
    #include "src/impl_arm_linux_or_android.c"
    #include "src/impl_mips_linux_or_android.c"
    #include "src/impl_ppc_linux.c"
    #include "src/impl_x86_freebsd.c"
    #include "src/impl_x86_linux_or_android.c"
    #include "src/impl_x86_windows.c"
    #include "src/impl_riscv_linux.c"
    #if GBX86
        #if (defined(__apple__) || defined(__APPLE__) || defined(__MACH__))
        // needed for src/impl_x86_macos.c:
        #define HAVE_SYSCTLBYNAME
        #endif
    #endif
    #include "src/impl_x86_macos.c"

#endif

