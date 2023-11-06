//------------------------------------------------------------------------------
// GB_masker_sparsity: determine the sparsity structure for C<M or !M>=Z
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determines the sparsity structure for R for computing R = masker (C,M,Z).
// If R should be hypersparse or sparse, on output, this function simply
// returns GxB_SPARSE.  The final determination is made by GB_add_phase0,
// as called by GB_masker.

// C is sparse or hypersparse on input.  It is never bitmap or full; that
// case is handled by GB_subassign instead.  R can be constructed as sparse,
// hypersparse, or bitmap (not full).  M and Z can have any sparsity pattern.

#include "GB_mask.h"

int GB_masker_sparsity      // return the sparsity structure for R
(
    // input:
    const GrB_Matrix C,     // input C matrix
    const GrB_Matrix M,     // mask for C, always present
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix Z      // input Z matrix
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity of R
    //--------------------------------------------------------------------------

    // In the tables below "sparse" means either sparse or hypersparse.

    ASSERT (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ;

    bool M_is_sparse = GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M) ;
    bool Z_is_sparse = GB_IS_SPARSE (Z) || GB_IS_HYPERSPARSE (Z) ;
    int R_sparsity ;

    if (Mask_comp)
    {

        //      ------------------------------------------
        //      C       <!M> =       Z              R
        //      ------------------------------------------

        //      sparse  sparse      sparse          sparse
        //      sparse  sparse      bitmap          bitmap
        //      sparse  sparse      full            bitmap

        //      sparse  bitmap      sparse          sparse
        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        sparse          sparse
        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        if (Z_is_sparse)
        { 
            R_sparsity = GxB_SPARSE ;
        }
        else
        { 
            R_sparsity = GxB_BITMAP ;
        }

    }
    else
    {

        //      ------------------------------------------
        //      C       <M> =        Z              R
        //      ------------------------------------------

        //      sparse  sparse      sparse          sparse
        //      sparse  sparse      bitmap          sparse
        //      sparse  sparse      full            sparse

        //      sparse  bitmap      sparse          sparse
        //      sparse  bitmap      bitmap          bitmap
        //      sparse  bitmap      full            bitmap

        //      sparse  full        sparse          sparse
        //      sparse  full        bitmap          bitmap
        //      sparse  full        full            bitmap

        if (M_is_sparse || Z_is_sparse)
        { 
            R_sparsity = GxB_SPARSE ;
        }
        else
        { 
            R_sparsity = GxB_BITMAP ;
        }
    }

    return (R_sparsity) ;
}

