//------------------------------------------------------------------------------
// GB_warnings.h: turn off compiler warnings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#if GB_COMPILER_ICC || GB_COMPILER_ICX

    //  10397: remark about where *.optrpt reports are placed
    //  15552: loop not vectorized
    #pragma warning (disable: 10397 15552 )

    // disable icc -w2 warnings
    //  191:  type qualifier meangingless
    //  193:  zero used for undefined #define
    #pragma warning (disable: 191 193 )

    // disable icc -w3 warnings
    //  144:  initialize with incompatible pointer
    //  181:  format
    //  869:  unused parameters
    //  1572: floating point compares
    //  1599: shadow
    //  2259: typecasting may lose bits
    //  161, 2282: unrecognized pragma
    //  2557: sign compare
    #pragma warning (disable: 161 144 181 869 1572 1599 2259 2282 2557 )

    // See GB_unused.h, for warnings 177 and 593, which are not globally
    // disabled, but selectively by #include'ing GB_unused.h as needed.

#elif GB_COMPILER_GCC

    // disable warnings for gcc 5.x and higher:
    #if ( __GNUC__ > 4 )
    // disable warnings
    #pragma GCC diagnostic ignored "-Wint-in-bool-context"
    #pragma GCC diagnostic ignored "-Wformat-truncation="
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
    #pragma GCC diagnostic ignored "-Wtype-limits"
    // enable these warnings as errors
    #pragma GCC diagnostic error "-Wmisleading-indentation"
    #endif

    // disable warnings from -Wall -Wextra -Wpendantic
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #if defined ( __cplusplus )
    #pragma GCC diagnostic ignored "-Wwrite-strings"
    #else
    #pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
    #endif

    // enable these warnings as errors
    #pragma GCC diagnostic error "-Wswitch-default"
    #if !defined ( __cplusplus )
    #pragma GCC diagnostic error "-Wmissing-prototypes"
    #endif

#elif GB_COMPILER_CLANG

    // disable warnings for clang
    #pragma clang diagnostic ignored "-Wpointer-sign"
    #pragma clang diagnostic ignored "-Wpass-failed"

#elif GB_COMPILER_MSC

    // disable MS Visual Studio warnings
    #pragma warning(disable:4146)

#endif

