#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/SuiteSparseAtomic.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2017-2024, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# determine if -latomic must be linked

include ( CheckCSourceCompiles )

set ( atomic_source
"
#if defined ( _MSC_VER ) && \
  !(defined ( __INTEL_CLANG_COMPILER ) || defined ( __INTEL_COMPILER ) || \
    defined ( __clang__ ) || defined ( __GNUC__ ))
    // MS Visual Studio compiler.  No need to check for atomics; GraphBLAS
    // uses MSC-specific methods.  (See Source/GB_atomics.h)
    int main (void) { return (0) ; }
#else
    // gcc, clang, icx, xlc, etc: see if -latomic is required
    #include <stdatomic.h>
    #include <stdint.h>
    int main (void)
    {
        _Atomic uint8_t t8 = 0 ;
        uint8_t e8 = 0, d8 = 0 ;
        atomic_compare_exchange_weak (&t8, &e8, d8) ;
        _Atomic uint16_t t16 = 0 ;
        uint16_t e16 = 0, d16 = 0 ;
        atomic_compare_exchange_weak (&t16, &e16, d16) ;
        _Atomic uint32_t t32 = 0 ;
        uint32_t e32 = 0, d32 = 0 ;
        atomic_compare_exchange_weak (&t32, &e32, d32) ;
        _Atomic uint64_t t64 = 0 ;
        uint64_t e64 = 0, d64 = 0 ;
        atomic_compare_exchange_weak (&t64, &e64, d64) ;
        return (0) ;
    }
#endif
" )

check_c_source_compiles ( "${atomic_source}" TEST_FOR_STDATOMIC )

if ( NOT TEST_FOR_STDATOMIC )
    # try with -latomic
    set ( CMAKE_REQUIRED_LIBRARIES "atomic" )
    check_c_source_compiles ( "${atomic_source}" TEST_FOR_STDATOMIC_WITH_LIBATOMIC )
    if ( NOT TEST_FOR_STDATOMIC_WITH_LIBATOMIC )
        # fails with -latomic
        message ( FATAL_ERROR "C11 atomics: failed" )
    endif ( )
    # source compiles but -latomic is required
    set ( LIBATOMIC_REQUIRED true )
    message ( STATUS "C11 atomics: OK, but -latomic required" )
else ( )
    set ( LIBATOMIC_REQUIRED false )
    message ( STATUS "C11 atomics: OK. -latomic not needed" )
endif ( )

