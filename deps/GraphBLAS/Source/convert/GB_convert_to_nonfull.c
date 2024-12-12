//------------------------------------------------------------------------------
// GB_convert_to_nonfull: ensure a matrix is not full (hyper, sparse, or bitmap)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The matrix A must be converted from full to any other sparsity structure.
// The full sparsity structure cannot tolerate the deletion of any entry but
// the other three can.

#include "GB.h"

GrB_Info GB_convert_to_nonfull      // ensure a matrix is not full
(
    GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_FULL (A)) ;

    //--------------------------------------------------------------------------
    // convert to bitmap, sparse, or hypersparse
    //--------------------------------------------------------------------------

    int sparsity_control = GB_sparsity_control (A->sparsity_control, A->vdim) ;

    if (sparsity_control & GxB_BITMAP)
    { 
        // C can become bitmap
        return (GB_convert_full_to_bitmap (A)) ;
    }
    else if (sparsity_control & GxB_SPARSE)
    { 
        // C can become sparse
        return (GB_convert_full_to_sparse (A)) ;
    }
    else if (sparsity_control & GxB_HYPERSPARSE)
    { 
        // C can become hypersparse
        return (GB_convert_any_to_hyper (A, Werk)) ;
    }
    else
    { 
        // none of the above conditions hold so make A bitmap
        return (GB_convert_full_to_bitmap (A)) ;
    }
}

