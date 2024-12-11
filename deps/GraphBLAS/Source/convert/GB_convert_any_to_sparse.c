//------------------------------------------------------------------------------
// GB_convert_any_to_sparse: convert any matrix to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#define GB_FREE_ALL ;

GrB_Info GB_convert_any_to_sparse // convert to sparse
(
    GrB_Matrix A,           // matrix to convert to sparse
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A being converted to sparse", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // convert A to sparse
    //--------------------------------------------------------------------------

    if (GB_IS_HYPERSPARSE (A))
    { 
        // convert from hypersparse to sparse
        GB_OK (GB_convert_hyper_to_sparse (A, true)) ;
    }
    else if (GB_IS_FULL (A))
    { 
        // convert from full to sparse
        GB_OK (GB_convert_full_to_sparse (A)) ;
    }
    else if (GB_IS_BITMAP (A))
    { 
        // convert from bitmap to sparse
        GB_OK (GB_convert_bitmap_to_sparse (A, Werk)) ;
    }
    else
    { 
        // already sparse; nothing to do
        ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to sparse", GB0) ;
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;
    return (GrB_SUCCESS) ;
}

