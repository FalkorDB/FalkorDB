//------------------------------------------------------------------------------
// GB_helper.h: helper functions for @GrB interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These functions are only used by the @GrB interface for
// SuiteSparse:GraphBLAS.

#ifndef GB_HELPER_H
#define GB_HELPER_H

#include "GB.h"

double GB_helper0 (void) ;

void GB_helper1              // convert zero-based indices to one-based
(
    double *restrict I_double,   // output array
    const GrB_Index *restrict I, // input array
    int64_t nvals                   // size of input and output arrays
) ;

void GB_helper1i             // convert zero-based indices to one-based
(
    int64_t *restrict I,         // input/output array
    int64_t nvals                   // size of input/output array
) ;

bool GB_helper3              // return true if OK, false on error
(
    int64_t *restrict List,      // size len, output array
    const double *restrict List_double, // size len, input array
    int64_t len,
    int64_t *List_max               // also compute the max entry in the list
) ;

bool GB_helper3i             // return true if OK, false on error
(
    int64_t *restrict List,      // size len, output array
    const int64_t *restrict List_int64, // size len, input array
    int64_t len,
    int64_t *List_max               // also compute the max entry in the list
) ;

bool GB_helper4              // return true if OK, false on error
(
    const GrB_Index *restrict I, // array of size len
    const int64_t len,
    GrB_Index *List_max             // find max (I) + 1
) ;

void GB_helper5              // construct pattern of S
(
    GrB_Index *restrict Si,         // array of size anz
    GrB_Index *restrict Sj,         // array of size anz
    const GrB_Index *restrict Mi,   // array of size mnz, M->i
    const GrB_Index *restrict Mj,   // array of size mnz
    const int64_t mvlen,               // M->vlen
    GrB_Index *restrict Ai,         // array of size anz, A->i
    const int64_t avlen,               // M->vlen
    const GrB_Index anz
) ;

void GB_helper7              // Kx = uint64 (0:mnz-1)
(
    uint64_t *restrict Kx,       // array of size mnz
    const GrB_Index mnz
) ;

void GB_helper8
(
    GB_void *C,         // output array of size nvals * s
    GB_void *A,         // input scalar of size s
    GrB_Index nvals,    // size of C
    size_t s            // size of each scalar
) ;

double GB_helper10       // norm (x-y,p), or -1 on error
(
    GB_void *x_arg,             // float or double, depending on type parameter
    bool x_iso,                 // true if x is iso
    GB_void *y_arg,             // same type as x, treat as zero if NULL
    bool y_iso,                 // true if x is iso
    GrB_Type type,              // GrB_FP32 or GrB_FP64
    int64_t p,                  // 0, 1, 2, INT64_MIN, or INT64_MAX
    GrB_Index n
) ;

void GB_make_shallow (GrB_Matrix A) ;

#endif

