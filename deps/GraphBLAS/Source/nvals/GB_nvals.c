//------------------------------------------------------------------------------
// GB_nvals: number of entries in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_nvals           // get the number of entries in a matrix
(
    GrB_Index *nvals,       // matrix has nvals entries
    const GrB_Matrix A,     // matrix to query
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (nvals) ;

    // leave zombies alone, and leave jumbled, but assemble any pending tuples
    GB_MATRIX_WAIT_IF_PENDING (A) ;

    //--------------------------------------------------------------------------
    // return the number of entries in the matrix
    //--------------------------------------------------------------------------

    // Pending tuples are disjoint from the zombies and the live entries in the
    // matrix.  However, there can be duplicates in the pending tuples, and the
    // number of duplicates has not yet been determined.  Thus, zombies can be
    // tolerated but pending tuples cannot.

    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    (*nvals) = GB_nnz (A) - (A->nzombies) ;
    return (GrB_SUCCESS) ;
}

