//------------------------------------------------------------------------------
// GB_helper.h: helper functions for @GrB interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These functions are only used by the @GrB MATLAB/Octave interface for
// SuiteSparse:GraphBLAS.

#ifndef GB_HELPER_H
#define GB_HELPER_H

#include "GB.h"
#include "math/GB_math.h"

void GB_helper5             // construct pattern of S
(
    // output:
    uint64_t *restrict Si,          // array of size anz
    uint64_t *restrict Sj,          // array of size anz
    // input:
    const void *Mi,                 // array of size mnz, M->i, may be NULL
    const bool Mi_is_32,            // if true, M->i is 32-bit; else 64-bit
    const uint64_t *restrict Mj,    // array of size mnz
    const int64_t mvlen,            // M->vlen
    const void *Ai,                 // array of size anz, A->i, may be NULL
    const bool Ai_is_32,            // if true, A->i is 32-bit; else 64-bit
    const int64_t avlen,            // A->vlen
    const uint64_t anz
) ;

void GB_helper7              // Kx = uint64 (0:mnz-1)
(
    uint64_t *restrict Kx,      // array of size mnz
    const uint64_t mnz
) ;

double GB_helper10       // norm (x-y,p), or -1 on error
(
    GB_void *x_arg,             // float or double, depending on type parameter
    bool x_iso,                 // true if x is iso
    GB_void *y_arg,             // same type as x, treat as zero if NULL
    bool y_iso,                 // true if x is iso
    GrB_Type type,              // GrB_FP32 or GrB_FP64
    int64_t p,                  // 0, 1, 2, INT64_MIN, or INT64_MAX
    uint64_t n
) ;

GxB_Container GB_helper_container (void) ;  // return the global Container
void GB_helper_container_new (void) ;       // allocate the global Container
void GB_helper_container_free (void) ;      // free the global Container

#endif

