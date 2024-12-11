//------------------------------------------------------------------------------
// GB_sparsity_control: ensure the sparsity_control is in the proper range
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

int GB_sparsity_control     // revised sparsity_control
(
    int sparsity_control,   // sparsity_control to be modified
    int64_t vdim            // A->vdim, or -1 to ignore this condition
)
{ 

    //--------------------------------------------------------------------------
    // ensure the sparsity control is in range 1 to 15
    //--------------------------------------------------------------------------

    sparsity_control = sparsity_control & GxB_ANY_SPARSITY ;
    if (sparsity_control == GxB_DEFAULT)
    { 
        // if zero, set to auto sparsity_control
        sparsity_control = GxB_AUTO_SPARSITY ;
    }

    //--------------------------------------------------------------------------
    // ensure vectors and scalars cannot become hypersparse
    //--------------------------------------------------------------------------

    if ((vdim == 0 || vdim == 1) && (sparsity_control & GxB_HYPERSPARSE))
    { 
        // a GrB_Scalar, GrB_Vector, or a GrB_Matrix with a single vector,
        // cannot be converted to hypersparse.  If the sparsity_control allows
        // for the hypersparse case, disable it and enable the sparse case
        // instead.
        sparsity_control = sparsity_control & (~GxB_HYPERSPARSE) ;
        sparsity_control = sparsity_control | GxB_SPARSE ;
    }

    //--------------------------------------------------------------------------
    // return revised sparsity_control
    //--------------------------------------------------------------------------

    return (sparsity_control) ;
}

