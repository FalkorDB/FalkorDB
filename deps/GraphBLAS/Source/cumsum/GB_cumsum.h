//------------------------------------------------------------------------------
// GB_cumsum.h: definitions for GB_cumsum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUMSUM_H
#define GB_CUMSUM_H

#include "include/GB_cumsum1.h"

bool GB_cumsum                  // cumulative sum of an array
(
    void *restrict count_arg,   // size n+1, input/output
    bool count_is_32,           // if true: count is uint32_t, else uint64_t
    const int64_t n,
    int64_t *restrict kresult,  // return k, if needed by the caller
    int nthreads,
    GB_Werk Werk
) ;

bool GB_cumsum_float            // cumulative sum of an array
(
    float *restrict count,      // size n+1, input/output
    const int64_t n,
    int nthreads,
    GB_Werk Werk
) ;

#endif

