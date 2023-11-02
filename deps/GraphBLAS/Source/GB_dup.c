//------------------------------------------------------------------------------
// GB_dup: make a deep copy of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, making a deep copy.  Not user-callable; this function does the work
// for user-callable functions GrB_*_dup.

// if numeric is false, C->x is allocated but not initialized.

// There is little use for the following feature, but (*Chandle) and A might be
// identical, with GrB_dup (&A, A).  The input matrix A will be lost, and will
// result in a memory leak, unless the user application does the following
// (which is valid and memory-leak free):

//  B = A ;

//  GrB_dup (&A, A) ;

//  GrB_free (&A) ;

//  GrB_free (&B) ;

// A is the new copy and B is the old copy.  Each should be freed when done.

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GB_dup             // make an exact copy of a matrix
(
    GrB_Matrix *Chandle,    // handle of output matrix to create
    const GrB_Matrix A,     // input matrix to copy
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Chandle != NULL) ;
    ASSERT_MATRIX_OK (A, "A to duplicate", GB0) ;
    (*Chandle) = NULL ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (A) ;

    //--------------------------------------------------------------------------
    // C = A
    //--------------------------------------------------------------------------

    // set C->iso = A->iso      OK
    GB_BURBLE_MATRIX (A, "(iso dup) ") ;
    return (GB_dup_worker (Chandle, A->iso, A, true, NULL)) ;
}

