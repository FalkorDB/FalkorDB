//------------------------------------------------------------------------------
// GB_bix_free: free A->(b,i,x) pending tuples, zombies; A->p, A->h unchanged
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Since A->p and A->h are unchanged, the matrix is still valid (unless it was
// invalid on input).  nnz(A) would report zero, and so would GrB_Matrix_nvals.

#include "GB_Pending.h"

void GB_bix_free                // free A->b, A->i, and A->x of a matrix
(
    GrB_Matrix A                // matrix with content to free
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 
        // nothing to do
        return ;
    }

    //--------------------------------------------------------------------------
    // free all but A->p and A->h
    //--------------------------------------------------------------------------

    // zombies and pending tuples are about to be deleted
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;

    // free A->b unless it is shallow
    if (!A->b_shallow)
    { 
        GB_FREE (&(A->b), A->b_size) ;
    }
    A->b = NULL ;
    A->b_size = 0 ;
    A->b_shallow = false ;

    // free A->i unless it is shallow
    if (!A->i_shallow)
    { 
        GB_FREE (&(A->i), A->i_size) ;
    }
    A->i = NULL ;
    A->i_size = 0 ;
    A->i_shallow = false ;

    // free A->x unless it is shallow
    if (!A->x_shallow)
    { 
        GB_FREE (&(A->x), A->x_size) ;
    }
    A->x = NULL ;               // GB_nnz_max (A) will report zero
    A->x_size = 0 ;
    A->x_shallow = false ;
    A->iso = false ;            // OK: all components are freed; no longer iso

    A->nvals = 0 ;

    // no zombies remain
    A->nzombies = 0 ;

    // an empty matrix is not jumbled
    A->jumbled = false ;

    // free the list of pending tuples
    GB_Pending_free (&(A->Pending)) ;
}

