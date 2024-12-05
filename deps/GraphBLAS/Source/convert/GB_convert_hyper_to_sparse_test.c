//------------------------------------------------------------------------------
// GB_convert_hyper_to_sparse_test: test conversion of hypersparse to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns true if a hypersparse matrix should be converted to sparse.
// Returns false if the matrix should stay hypersparse.

// A matrix with vdim <= 1 must always be sparse, not hypersparse;
// that is, a GrB_Vector is never hypersparse.

#include "GB.h"

bool GB_convert_hyper_to_sparse_test    // test for hypersparse to sparse
(
    float hyper_switch,     // A->hyper_switch
    int64_t k,              // # of non-empty vectors of A (estimate is OK)
    int64_t vdim            // A->vdim
)
{ 

    // get the vector dimension of this matrix
    float n = (float) vdim ;

    // ensure k is in the range 0 to n, inclusive
    k = GB_IMAX (k, 0) ;
    k = GB_IMIN (k, n) ;

    return (n <= 1 || (((float) k) > n * hyper_switch * 2)) ;
}

