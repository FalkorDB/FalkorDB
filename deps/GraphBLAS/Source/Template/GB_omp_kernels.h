//------------------------------------------------------------------------------
// GB_omp_kernels.h: definitions using OpenMP in SuiteSparse:GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_OMP_KERNELS_H
#define GB_OMP_KERNELS_H

//------------------------------------------------------------------------------
// determine the OpenMP version
//------------------------------------------------------------------------------

#if GB_COMPILER_MSC

    // MS Visual Studio supports OpenMP 2.0, and does not have the atomic
    // capture clause.  However, it has interlocked compare/exchange functions
    // that are used instead (see GB_atomics.h).
    #include <intrin.h>

#elif defined ( _OPENMP )

    // All other compilers must either support OpenMP 3.1 or later, or not use
    // OpenMP at all.
    #if _OPENMP < 201107
        #error "OpenMP 3.1 or later required (recompile without OpenMP)"
    #endif

#endif

//------------------------------------------------------------------------------
// OpenMP include file and definitions
//------------------------------------------------------------------------------

#if defined ( _OPENMP )
    #include <omp.h>
    #define GB_OPENMP_MAX_THREADS       omp_get_max_threads ( )
    #define GB_OPENMP_GET_WTIME         omp_get_wtime ( )
#else
    #define GB_OPENMP_MAX_THREADS       (1)
    #define GB_OPENMP_GET_WTIME         (0)
#endif

#endif

