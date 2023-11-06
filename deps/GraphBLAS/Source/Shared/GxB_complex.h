//------------------------------------------------------------------------------
// GxB_complex.h: definitions for the GraphBLAS complex types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If a user-defined operator needs access to the GraphBLAS complex types,
// GxB_FC32_t and GxB_FC64_t types, then this file can be #include'd in the
// defining string.  See Demo/Include/usercomplex.h for an example.

// See:
// https://www.drdobbs.com/complex-arithmetic-in-the-intersection-o/184401628#

#ifndef GXB_COMPLEX_H
#define GXB_COMPLEX_H

    #if ( _MSC_VER && !(__INTEL_COMPILER || __INTEL_CLANG_COMPILER) )

        // Microsoft Windows complex types for C
        #include <complex.h>
        #undef I
        typedef _Fcomplex GxB_FC32_t ;
        typedef _Dcomplex GxB_FC64_t ;
        #define GxB_CMPLXF(r,i) (_FCbuild (r,i))
        #define GxB_CMPLX(r,i)  ( _Cbuild (r,i))
        #define GB_HAS_CMPLX_MACROS 1

    #else

        // C11 complex types
        #include <complex.h>
        #undef I
        typedef float  _Complex GxB_FC32_t ;
        typedef double _Complex GxB_FC64_t ;
        #if (defined (CMPLX) && defined (CMPLXF))
            // use the C11 CMPLX and CMPLXF macros
            #define GxB_CMPLX(r,i) CMPLX (r,i)
            #define GxB_CMPLXF(r,i) CMPLXF (r,i)
            #define GB_HAS_CMPLX_MACROS 1
        #else
            // gcc 6.2 on the the Mac doesn't #define CMPLX
            #define GB_HAS_CMPLX_MACROS 0
            #define GxB_CMPLX(r,i) \
            ((GxB_FC64_t)((double)(r)) + (GxB_FC64_t)((double)(i) * _Complex_I))
            #define GxB_CMPLXF(r,i) \
            ((GxB_FC32_t)((float)(r)) + (GxB_FC32_t)((float)(i) * _Complex_I))
        #endif
    #endif
#endif

