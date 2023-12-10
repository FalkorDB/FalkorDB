//------------------------------------------------------------------------------
// GB_hyper_hash_free: free the A->Y hyper_hash of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Free the A->Y hyper_hash of a matrix.  The matrix remains valid since
// the hyper_hash can be reconstructed by any method that requires it,
// via GB_hyper_hash_build.

#include "GB.h"

void GB_hyper_hash_free         // free the A->Y hyper_hash of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{

    //--------------------------------------------------------------------------
    // free A->Y
    //--------------------------------------------------------------------------

    if (A != NULL)
    { 
        if (!A->Y_shallow)
        { 
            GB_Matrix_free (&(A->Y)) ;
        }
        A->Y = NULL ;
        A->Y_shallow = false ;
    }
}

