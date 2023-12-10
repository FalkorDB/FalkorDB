//------------------------------------------------------------------------------
// GB_hyper_hash_need: determine if a matrix needs its hyper_hash built
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A hypersparse matrix A can use a hyper_hash matrix A->Y to speed up access
// to its hyper list, A->h.  For extremely sparse matrices, building the
// hyper_hash can be costly, however.

// If A has pending work, this test is not useful, since the format of A can
// change, and A->nvec could change if zombies are removed and pending tuples
// assembled.  This test should only be used if a matrix has no other pending
// work.

#include "GB.h"

bool GB_hyper_hash_need
(
    GrB_Matrix A
)
{

    if (A == NULL || !GB_IS_HYPERSPARSE (A))
    { 
        // only hypersparse matrices require a hyper_hash
        return (false) ;
    }

    if (A->Y != NULL)
    { 
        // A already has a hyper_hash
        return (false) ;
    }

    // A is hypersparse, and has no hyper_hash.  Check how many non-empty
    // vectors it has.  A->Y should be built if A has a significant number of
    // non-empty vectors.

    // FUTURE: make this also a per-matrix parameter for GrB_get/set

    int64_t hyper_hash = GB_Global_hyper_hash_get ( ) ;

    return (A->nvec > hyper_hash) ;
}

