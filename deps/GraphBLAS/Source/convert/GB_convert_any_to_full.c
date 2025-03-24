//------------------------------------------------------------------------------
// GB_convert_any_to_full: convert any matrix to full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All entries in A must be present, with no pending work; GB_as_if_full (A)
// must be true on input, or A must be iso.  A may be hypersparse, sparse,
// bitmap, or full on input. A is full on output.  If A is iso, it remains so
// on output.

#include "GB.h"

void GB_convert_any_to_full     // convert any matrix to full
(
    GrB_Matrix A                // matrix to convert to full
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting any to full", GB0) ;
    ASSERT (A->iso || GB_as_if_full (A)) ;

    if (GB_IS_FULL (A))
    { 
        // already full; nothing to do
        return ;
    }

    GB_BURBLE_N (A->nvals, "(%s to full) ", (A->h != NULL) ? "hypersparse" :
        (GB_IS_BITMAP (A) ? "bitmap" : "sparse")) ;

    //--------------------------------------------------------------------------
    // free A->h, A->p, A->i, and A->b
    //--------------------------------------------------------------------------

    GB_phy_free (A) ;

    if (!A->i_shallow) GB_FREE_MEMORY (&(A->i), A->i_size) ;
    A->i = NULL ;
    A->i_shallow = false ;
    A->p_is_32 = false ;    // OK: full always has p_is_32 = false
    A->j_is_32 = false ;    // OK: full always has j_is_32 = false
    A->i_is_32 = false ;    // OK: full always has i_is_32 = false

    if (!A->b_shallow) GB_FREE_MEMORY (&(A->b), A->b_size) ;
    A->b = NULL ;
    A->b_shallow = false ;

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;

    A->plen = -1 ;
    A->nvec = avdim ;
//  A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;
    GB_nvec_nonempty_set (A, (avlen == 0) ? 0 : avdim) ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from any to full", GB0) ;
    ASSERT (GB_IS_FULL (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
}

