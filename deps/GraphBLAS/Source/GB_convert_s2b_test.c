//------------------------------------------------------------------------------
// GB_convert_s2b_test: test conversion of hyper/sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if a sparse or hypersparse matrix should be converted to bitmap
// (or full if all entries are present, but in that case, this function is not
// called).  Returns false if the matrix should stay hypersparse/sparse.

// See GB_convert_bitmap_to_sparse_test for a description of this rule.
// These default rules may change in future releases of SuiteSparse:GraphBLAS.

// This rule is not used if all entries are present.  In that case, the matrix
// becomes full, not bitmap, assuming the full format permitted by the sparsity
// control setting of the matrix.

#include "GB.h"

bool GB_convert_s2b_test    // test for hyper/sparse to bitmap
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

    // A should switch to bitmap if the following condition is true:
    return (nnz > bitmap_switch * nnz_dense && nnz_dense < (float) GB_NMAX) ;
}

