#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/GraphBLAS_complex.cmake
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2024-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# Check for C compiler support for complex floating point numbers and used API

include ( CheckSourceCompiles )

# Check for C99 complex number arithmetic

check_source_compiles ( C
    "#include <complex.h>
    int main (void)
    {
        double _Complex z1 = 1.0 ;
        double _Complex z2 = 1.0 * I ;
        double _Complex z3 = z1 * z2 ;
        return ((int) creal (z3)) ;
    }"
    GxB_HAVE_COMPLEX_C99 )

if ( NOT GxB_HAVE_COMPLEX_C99 )
    # Check for complex number arithmetic as implemented by MSVC

    check_source_compiles ( C
        "#include <complex.h>
        int main (void)
        {
            _Dcomplex z1 = {1., 0.} ;
            _Dcomplex z2 = {0., 1.} ;
            _Dcomplex z3 = _Cmulcc(z1, z2) ;
            return 0 ;
        }"
        GxB_HAVE_COMPLEX_MSVC )
endif ( )

if ( NOT GxB_HAVE_COMPLEX_C99 AND NOT GxB_HAVE_COMPLEX_MSVC )
    message ( FATAL_ERROR "The selected C compiler does not support complex floating point numbers; try another compiler." )
endif ( )
