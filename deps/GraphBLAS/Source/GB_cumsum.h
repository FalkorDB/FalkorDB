//------------------------------------------------------------------------------
// GB_cumsum.h: definitions for GB_cumsum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUMSUM_H
#define GB_CUMSUM_H

void GB_cumsum                      // cumulative sum of an array
(
    int64_t *restrict count,     // size n+1, input/output
    const int64_t n,
    int64_t *restrict kresult,   // return k, if needed by the caller
    int nthreads,
    GB_Werk Werk
) ;

#endif

