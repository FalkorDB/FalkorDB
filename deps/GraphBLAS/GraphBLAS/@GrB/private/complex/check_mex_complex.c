//------------------------------------------------------------------------------
// check_mex_complex: test the complex type supported by the compiler
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "mex.h"

#if defined (GxB_HAVE_COMPLEX_MSVC)
    // Microsoft Windows complex types for C
    #include <complex.h>
    #include <math.h>
    typedef _Fcomplex GxB_FC32_t ;
    typedef _Dcomplex GxB_FC64_t ;
#elif defined (GxB_HAVE_COMPLEX_C99)
    // C11 complex types
    #include <complex.h>
    #include <math.h>
    typedef float  _Complex GxB_FC32_t ;
    typedef double _Complex GxB_FC64_t ;
#else
    #error "Complex type undefined"
#endif

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    #if defined (GxB_HAVE_COMPLEX_MSVC)
    _Dcomplex z1 = {2., 3.} ;
    _Dcomplex z2 = {1., 2.} ;
    _Dcomplex z3 = _Cmulcc (z1, z2) ;
    mexPrintf ("MSVC complex type OK: (%g,%g)\n", creal (z3), cimag (z3)) ;
    #elif defined (GxB_HAVE_COMPLEX_C99)
    double _Complex z1 = 2. + 3.*I ;
    double _Complex z2 = 1. + 2.*I ;
    double _Complex z3 = z1 * z2 ;
    mexPrintf ("C99 complex type OK: (%g,%g)\n", creal (z3), cimag (z3)) ;
    #endif
}

