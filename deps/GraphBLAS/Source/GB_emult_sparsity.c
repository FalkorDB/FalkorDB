//------------------------------------------------------------------------------
// GB_emult_sparsity: determine the sparsity structure for C<M or !M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determines the sparsity structure for C, for computing C=A.*B, C<M>=A.*B,
// or C<!M>=A.*B, based on the sparsity structures of M, A, and B, and whether
// or not M is complemented.  It also decides if the mask M should be applied
// by GB_emult, or if C=A.*B should be computed without the mask, and the mask
// applied later.

// If C should be constructed as hypersparse or sparse, this function simply
// returns GxB_SPARSE.  The final determination is made later.

// If both A and B are full, then GB_ewise can call GB_add instead of GB_emult.
// This is the only case where the eWise multiply can produce a full C matrix,
// and as a result, there is no need for a GB_emult to handle the case when
// C is full.

#include "GB_emult.h"
#include "GB_mask_very_sparse.h"

int GB_emult_sparsity       // return the sparsity structure for C
(
    // output:
    bool *apply_mask,       // if true then mask will be applied by GB_emult
    int *ewise_method,      // method to use
    // input:
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B      // input B matrix
)
{

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    // Unless deciding otherwise, use the mask if it appears
    (*apply_mask) = (M != NULL) ;
    int C_sparsity ;

    // In the table below, sparse/hypersparse are listed as "sparse".  If C is
    // listed as sparse, it will become sparse or hypersparse, depending on the
    // method.

    bool M_is_sparse_or_hyper = GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M) ;
    bool A_is_sparse_or_hyper = GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A) ;
    bool B_is_sparse_or_hyper = GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B) ;

    bool A_is_full = GB_IS_FULL (A) ;
    bool B_is_full = GB_IS_FULL (B) ;

    if (M == NULL)
    {

        //      ------------------------------------------
        //      C       =           A       .*      B
        //      ------------------------------------------
        //      sparse  .           sparse          sparse  (method: 8)
        //      sparse  .           sparse          bitmap  (method: 2)
        //      sparse  .           sparse          full    (method: 2)
        //      sparse  .           bitmap          sparse  (method: 3)
        //      bitmap  .           bitmap          bitmap  (method: 5)
        //      bitmap  .           bitmap          full    (method: 5)
        //      sparse  .           full            sparse  (method: 3)
        //      bitmap  .           full            bitmap  (method: 5)
        //      full    .           full            full    (must use GB_add)

        if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
        { 
            // C=A.*B with A and B both sparse/hyper, C sparse
            C_sparsity = GxB_SPARSE ;
            (*ewise_method) = GB_EMULT_METHOD8 ;
        }
        else if (A_is_sparse_or_hyper)
        { 
            // C=A.*B with A sparse/hyper, B bitmap/full
            C_sparsity = GxB_SPARSE ;
            (*ewise_method) = GB_EMULT_METHOD2 ;
        }
        else if (B_is_sparse_or_hyper)
        { 
            // C=A.*B with B sparse/hyper, A bitmap/full
            C_sparsity = GxB_SPARSE ;
            (*ewise_method) = GB_EMULT_METHOD3 ;
        }
        else if (A_is_full && B_is_full)
        { 
            // C=A.*B with A and B full, must use GB_add since GB_emult does
            // not handle the case when C is full.
            C_sparsity = GxB_FULL ;
            (*ewise_method) = GB_EMULT_METHOD1_ADD ;
        }
        else
        { 
            // C=A.*B, otherwise, C bitmap
            C_sparsity = GxB_BITMAP ;
            (*ewise_method) = GB_EMULT_METHOD5 ;
        }

    }
    else if (!Mask_comp)
    {

        if (M_is_sparse_or_hyper)
        { 

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (method: 8)
            //      sparse  sparse      sparse          bitmap  (9 or 2)
            //      sparse  sparse      sparse          full    (9 or 2)
            //      sparse  sparse      bitmap          sparse  (10 or 3)
            //      sparse  sparse      bitmap          bitmap  (method: 4)
            //      sparse  sparse      bitmap          full    (method: 4)
            //      sparse  sparse      full            sparse  (10 or 3)
            //      sparse  sparse      full            bitmap  (method: 4)
            //      sparse  sparse      full            full    (GB_add or 4)

            // C<M>=A.*B with M sparse/hyper, C sparse
            C_sparsity = GxB_SPARSE ;

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            { 
                // C<M>=A.*B with A and B both sparse/hyper, C sparse
                // apply the mask only if it is extremely sparse
                (*apply_mask) = GB_MASK_VERY_SPARSE (8, M, A, B) ;
                (*ewise_method) = GB_EMULT_METHOD8 ;
            }
            else if (A_is_sparse_or_hyper)
            {
                // C<M>=A.*B with A sparse/hyper, B bitmap/full
                // apply the mask only if it is very sparse
                if (GB_MASK_VERY_SPARSE (2, M, A, B))
                { 
                    // C<M>=A.*B with A sparse/hyper, B bitmap/full
                    (*apply_mask) = true ;
                    (*ewise_method) = GB_EMULT_METHOD9 ;
                }
                else
                { 
                    // C<M>=A.*B with A sparse/hyper, B bitmap/full, mask later
                    (*apply_mask) = false ;
                    (*ewise_method) = GB_EMULT_METHOD2 ;
                }
            }
            else if (B_is_sparse_or_hyper)
            {
                // C<M>=A.*B with B sparse/hyper, A bitmap/full
                // apply the mask only if it is very sparse
                if (GB_MASK_VERY_SPARSE (2, M, A, B))
                { 
                    // C<M>=A.*B with A bitmap/full, B sparse/hyper
                    (*apply_mask) = true ;
                    (*ewise_method) = GB_EMULT_METHOD10 ;
                }
                else
                { 
                    // C<M>=A.*B with A bitmap/full, B sparse/hyper, mask later
                    (*apply_mask) = false ;
                    (*ewise_method) = GB_EMULT_METHOD3 ;
                }
            }
            else if (A_is_full && B_is_full)
            { 
                // C=A.*B with A and B full
                // (*ewise_method) = GB_EMULT_METHOD1_ADD ;  this is possible
                (*ewise_method) = GB_EMULT_METHOD4 ;
            }
            else
            { 
                // C=A.*B, otherwise
                (*ewise_method) = GB_EMULT_METHOD4 ;
            }

        }
        else
        {

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          sparse  (method: 8)
            //      sparse  bitmap      sparse          bitmap  (method: 2)

            //      sparse  bitmap      sparse          full    (method: 2)
            //      sparse  bitmap      bitmap          sparse  (method: 3)
            //      bitmap  bitmap      bitmap          bitmap  (method: 7)
            //      bitmap  bitmap      bitmap          full    (method: 7)
            //      sparse  bitmap      full            sparse  (method: 3)
            //      bitmap  bitmap      full            bitmap  (method: 7)
            //      bitmap  bitmap      full            full    (GB_add or 7)

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      sparse  full        sparse          sparse  (method: 8)
            //      sparse  full        sparse          bitmap  (method: 2)
            //      sparse  full        sparse          full    (method: 2)
            //      sparse  full        bitmap          sparse  (method: 3)
            //      bitmap  full        bitmap          bitmap  (method: 7)
            //      bitmap  full        bitmap          full    (method: 7)
            //      sparse  full        full            sparse  (method: 3)
            //      bitmap  full        full            bitmap  (method: 7)
            //      bitmap  full        full            full    (GB_add or 7)

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            { 
                // C<M>=A.*B with A and B both sparse/hyper, C sparse
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD8 ;
            }
            else if (A_is_sparse_or_hyper)
            { 
                // C<M>=A.*B with A sparse/hyper, B bitmap/full
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD2 ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C<M>=A.*B with B sparse/hyper, A bitmap/full
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD3 ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C<M>=A.*B with A and B full
                C_sparsity = GxB_BITMAP ;
                // (*ewise_method) = GB_EMULT_METHOD1_ADD ;  this is possible
                (*ewise_method) = GB_EMULT_METHOD7 ;
            }
            else
            { 
                // C=A.*B, otherwise, C bitmap
                C_sparsity = GxB_BITMAP ;
                (*ewise_method) = GB_EMULT_METHOD7 ;
            }
        }

    }
    else // Mask_comp
    {

        if (M_is_sparse_or_hyper)
        {

            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (8: M later)
            //      sparse  sparse      sparse          bitmap  (2: M later)
            //      sparse  sparse      sparse          full    (2: M later)
            //      sparse  sparse      bitmap          sparse  (3: M later)
            //      bitmap  sparse      bitmap          bitmap  (method: 6)
            //      bitmap  sparse      bitmap          full    (method: 6)
            //      sparse  sparse      full            sparse  (3: M later)
            //      bitmap  sparse      full            bitmap  (method: 6)
            //      bitmap  sparse      full            full    (GB_add or 6)

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            { 
                // C<!M>=A.*B with A and B sparse/hyper, C sparse, M later
                (*apply_mask) = false ;
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD8 ;
            }
            else if (A_is_sparse_or_hyper)
            { 
                // C<!M>=A.*B with A sparse/hyper, B bitmap/full, M later
                (*apply_mask) = false ;
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD2 ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C<!M>=A.*B with B sparse/hyper, A bitmap/full, M later
                (*apply_mask) = false ;
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD3 ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C<!M>=A.*B with A and B full
                C_sparsity = GxB_BITMAP ;
                // (*ewise_method) = GB_EMULT_METHOD1_ADD ;  this is possible
                (*ewise_method) = GB_EMULT_METHOD6 ;
            }
            else
            { 
                // C<!M>=A.*B, otherwise, C bitmap
                C_sparsity = GxB_BITMAP ;
                (*ewise_method) = GB_EMULT_METHOD6 ;
            }

        }
        else
        {

            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          sparse  (method: 8)
            //      sparse  bitmap      sparse          bitmap  (method: 2)
            //      sparse  bitmap      sparse          full    (method: 2)
            //      sparse  bitmap      bitmap          sparse  (method: 3)
            //      bitmap  bitmap      bitmap          bitmap  (method: 7)
            //      bitmap  bitmap      bitmap          full    (method: 7)
            //      sparse  bitmap      full            sparse  (method: 3)
            //      bitmap  bitmap      full            bitmap  (method: 7)
            //      bitmap  bitmap      full            full    (GB_add or 7)

            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  full        sparse          sparse  (method: 8)
            //      sparse  full        sparse          bitmap  (method: 2)
            //      sparse  full        sparse          full    (method: 2)
            //      sparse  full        bitmap          sparse  (method: 3)
            //      bitmap  full        bitmap          bitmap  (method: 7)
            //      bitmap  full        bitmap          full    (method: 7)
            //      sparse  full        full            sparse  (method: 3)
            //      bitmap  full        full            bitmap  (method: 7)
            //      bitmap  full        full            full    (GB_add or 7)

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            { 
                // C<!M>=A.*B with A and B both sparse/hyper, C sparse
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD8 ;
            }
            else if (A_is_sparse_or_hyper)
            { 
                // C<!M>=A.*B with A sparse/hyper, B bitmap/full
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD2 ;
            }
            else if (B_is_sparse_or_hyper)
            { 
                // C<!M>=A.*B with B sparse/hyper, A bitmap/full
                C_sparsity = GxB_SPARSE ;
                (*ewise_method) = GB_EMULT_METHOD3 ;
            }
            else if (A_is_full && B_is_full)
            { 
                // C<!M>=A.*B with A and B full
                C_sparsity = GxB_BITMAP ;
                // (*ewise_method) = GB_EMULT_METHOD1_ADD ; this is possible
                (*ewise_method) = GB_EMULT_METHOD7 ;
            }
            else
            { 
                // C<!M>=A.*B, otherwise, C bitmap
                C_sparsity = GxB_BITMAP ;
                (*ewise_method) = GB_EMULT_METHOD7 ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (C_sparsity) ;
}

