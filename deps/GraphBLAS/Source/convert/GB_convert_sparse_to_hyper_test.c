//------------------------------------------------------------------------------
// GB_convert_sparse_to_hyper_test: test for sparse to hypersparse conversion
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if a sparse matrix should be converted to hypersparse.
// Returns false if the matrix should stay sparse.

// A matrix with vdim <= 1 must always be sparse, not hypersparse;
// that is, a GrB_Vector is never hypersparse.

#include "GB.h"

bool GB_convert_sparse_to_hyper_test  // test sparse to hypersparse conversion
(
    float hyper_switch,     // A->hyper_switch
    int64_t k,              // # of non-empty vectors of A (an estimate is OK)
    int64_t vdim            // A->vdim
)
{ 

    // get the vector dimension of this matrix
    float n = (float) vdim ;

    // ensure k is in the range 0 to n, inclusive
    k = GB_IMAX (k, 0) ;
    k = GB_IMIN (k, n) ;

    return (n > 1 && (((float) k) <= n * hyper_switch)) ;
}

