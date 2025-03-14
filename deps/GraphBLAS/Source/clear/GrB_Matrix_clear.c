//------------------------------------------------------------------------------
// GrB_Matrix_clear: clears the content of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The A->x and A->i content is freed and the vector pointers A->p are set to
// zero.  This puts the matrix A in the same state it had after GrB_Matrix_new
// (&A, ...).  The dimensions and type of A are not changed.  The matrix A on
// input may have readonly (shallow) components; these are simply removed.

#include "GB.h"

GrB_Info GrB_Matrix_clear   // clear a matrix of all entries;
(                           // type and dimensions remain unchanged
    GrB_Matrix A            // matrix to clear
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_WHERE_1 (A, "GrB_Matrix_clear (A)") ;

    //--------------------------------------------------------------------------
    // clear the matrix
    //--------------------------------------------------------------------------

    return (GB_clear (A, Werk)) ;
}

