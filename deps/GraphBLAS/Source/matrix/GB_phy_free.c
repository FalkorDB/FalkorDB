//------------------------------------------------------------------------------
// GB_phy_free: free the A->p, A->h, and A->Y content of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Free the A->p, A->h, and A->Y content of a matrix.  The matrix becomes
// invalid, and would generate a GrB_INVALID_OBJECT error if passed to a
// user-callable GraphBLAS function.

// A->p_is_32, A->j_is_32, and A->i_is_32 are unchanged.

#include "GB.h"

void GB_phy_free                // free A->p, A->h, and A->Y of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 
        return ;
    }

    //--------------------------------------------------------------------------
    // free A->p, A->h, and A->Y
    //--------------------------------------------------------------------------

    // free A->p unless it is shallow
    if (!A->p_shallow)
    { 
        GB_FREE_MEMORY (&(A->p), A->p_size) ;
    }
    A->p = NULL ;
    A->p_size = 0 ;
    A->p_shallow = false ;

    // free A->h and A->Y
    GB_hy_free (A) ;

    A->plen = 0 ;
    A->nvec = 0 ;
    A->nvals = 0 ;
//  A->nvec_nonempty = 0 ;
    GB_nvec_nonempty_set (A, 0) ;

    //--------------------------------------------------------------------------
    // set the status to invalid
    //--------------------------------------------------------------------------

    // If this matrix is used as input to a user-callable GraphBLAS function,
    // it will generate an error: GrB_INVALID_OBJECT.
    A->magic = GB_MAGIC2 ;
}

