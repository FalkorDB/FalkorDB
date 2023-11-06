//------------------------------------------------------------------------------
// GB_convert_bitmap_to_sparse_test: test conversion of bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if a bitmap matrix should be converted to sparse.
// Returns false if the matrix should stay bitmap.

// If A is m-by-n and A->sparsity_control is GxB_ANY_SPARSITY with b =
// A->bitmap_switch, the matrix switches to bitmap if nnz(A)/(m*n) > b.  A
// bitmap matrix switches to sparse if nnz(A)/(m*n) <= b/2.  A matrix whose
// density is between b/2 and b remains in its current state.

// A->bitmap_switch is normally a fraction in range 0 to 1.  A value of 1
// ensures that A is never converted to bitmap.

// These default rules may change in future releases of SuiteSparse:GraphBLAS.

// If this test returns true and the matrix changes to sparse, then the rule
// for A->hyper_switch may then convert it from sparse to hypersparse.

#include "GB.h"

bool GB_convert_bitmap_to_sparse_test    // test for hyper/sparse to bitmap
(
    float bitmap_switch,    // A->bitmap_switch
    int64_t anz,            // # of entries in A = GB_nnz (A)
    int64_t vlen,           // A->vlen
    int64_t vdim            // A->vdim
)
{ 

    // current number of entries in the matrix or vector
    float nnz = (float) anz ;

    // maximum number of entries in the matrix or vector
    float nnz_dense = ((float) vlen) * ((float) vdim) ;

    // A should switch to sparse if the following condition is true:
    return (nnz <= (bitmap_switch/2) * nnz_dense) ;
}

