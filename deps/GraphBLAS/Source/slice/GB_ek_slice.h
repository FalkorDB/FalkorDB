//------------------------------------------------------------------------------
// GB_ek_slice.h: slice the entries and vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EK_SLICE_H
#define GB_EK_SLICE_H

#include "GB.h"

//------------------------------------------------------------------------------
// GB_ek_slice prototypes
//------------------------------------------------------------------------------

// Slice the entries of a matrix or vector into ntasks slices.

// Task t does entries pstart_slice [t] to pstart_slice [t+1]-1 and
// vectors kfirst_slice [t] to klast_slice [t].  The first and last vectors
// may be shared with prior slices and subsequent slices.

// On input, ntasks must be <= nnz (A), unless nnz (A) is zero.  In that
// case, ntasks must be 1.

void GB_ek_slice_merge2     // merge final results for matrix C
(
    // output
    int64_t *C_nvec_nonempty,       // # of non-empty vectors in C
    int64_t *restrict Cp_kfirst,    // size ntasks
    // input/output
    int64_t *restrict Cp,           // size cnvec+1
    // input
    const int64_t cnvec,
    const int64_t *restrict Wfirst, // size ntasks
    const int64_t *restrict Wlast,  // size ntasks
    const int64_t *A_ek_slicing,    // size 3*ntasks+1
    const int ntasks,               // # of tasks used to construct C
    const int nthreads,             // # of threads to use
    GB_Werk Werk
) ;

#endif
