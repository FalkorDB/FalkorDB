//------------------------------------------------------------------------------
// GB_convert_any_to_bitmap: convert any matrix to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input matrix may be jumbled and have zombies, and can still be converted
// to a bitmap.

#include "GB.h"
#define GB_FREE_ALL ;

GrB_Info GB_convert_any_to_bitmap   // convert to bitmap
(
    GrB_Matrix A,           // matrix to convert to bitmap
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A being converted to bitmap", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;    // A can have zombies
    ASSERT (GB_JUMBLED_OK (A)) ;    // A can be jumbled
    GB_MATRIX_WAIT_IF_PENDING (A) ; // A cannot have pending tuples
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // convert A to bitmap
    //--------------------------------------------------------------------------

    if (GB_IS_HYPERSPARSE (A))
    { 
        // convert from hypersparse to bitmap
        GB_OK (GB_convert_s2b (A, Werk)) ;
    }
    else if (GB_IS_FULL (A))
    { 
        // convert from full to bitmap
        GB_OK (GB_convert_full_to_bitmap (A)) ;
    }
    else if (GB_IS_BITMAP (A))
    { 
        // already bitmap; nothing to do
        ;
    }
    else
    { 
        // convert from sparse to bitmap
        GB_OK (GB_convert_s2b (A, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to bitmap", GB0) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

