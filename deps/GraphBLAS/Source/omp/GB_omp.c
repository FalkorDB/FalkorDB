//------------------------------------------------------------------------------
// GB_omp.c: wrappers for OpenMP functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

int GB_omp_get_max_threads (void)
{ 
    return (GB_OPENMP_MAX_THREADS) ;
}

double GB_omp_get_wtime (void)
{ 
    return (GB_OPENMP_GET_WTIME) ;
}

