//------------------------------------------------------------------------------
// GB_transpose_in_place: in-place transpose (to change A->is_csc format)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All other uses of GB_transpose are not in-place.
// No operator is applied and no typecasting is done.
// Does nothing if A is already in the requested format.

#include "transpose/GB_transpose.h"
#define GB_FREE_ALL ;

GrB_Info GB_transpose_in_place  // A=A', to change A->is_csc
(
    GrB_Matrix A,           // input/output matrix
    const bool new_csc,     // desired format, by row (false) or by col (true)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // conform the matrix to the new by-row/by-col format
    //--------------------------------------------------------------------------

    if (A->is_csc != new_csc)
    { 

        //----------------------------------------------------------------------
        // check inputs
        //----------------------------------------------------------------------

        GrB_Info info ;
        ASSERT_MATRIX_OK (A, "A for new csc format", GB0) ;
        ASSERT (GB_PENDING_OK (A)) ;
        ASSERT (GB_ZOMBIES_OK (A)) ;
        ASSERT (GB_JUMBLED_OK (A)) ;
        GB_BURBLE_N (GB_nnz (A), "(transpose to set orientation) ") ;

        //----------------------------------------------------------------------
        // swap A->[ji]_control
        //----------------------------------------------------------------------

        int8_t j_control = A->j_control ;
        A->j_control = A->i_control ;
        A->i_control = j_control ;

        //----------------------------------------------------------------------
        // in-place A = A', with the new csc setting
        //----------------------------------------------------------------------

        GB_OK (GB_transpose (A,
            /* no change of type: */ NULL,
            /* new requested format: */ new_csc,
            /* in-place transpose: */ A,
            /* no operator: */ NULL, NULL, false, false,
            Werk)) ;

        ASSERT_MATRIX_OK (A, "A with new csc format", GB0) ;
        ASSERT (A->is_csc == new_csc) ;
        ASSERT (!GB_PENDING (A)) ;
        ASSERT (!GB_ZOMBIES (A)) ;
        ASSERT (GB_JUMBLED_OK (A)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

