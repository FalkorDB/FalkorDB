//------------------------------------------------------------------------------
// GB_add_sparsity: determine the sparsity structure for C<M or !M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determines the sparsity structure for C, for computing C=A+B, C<M>=A+B,
// or C<!M>=A+B, based on the sparsity structures of M, A, and B, and whether
// or not M is complemented.  It also decides if the mask M should be applied
// by GB_add, or if C=A+B should be computed without the mask, and the mask
// applied later.

// If C should be hypersparse or sparse, on output, this function simply
// returns GxB_SPARSE.  The final determination is made by GB_add_phase0.

#include "GB_add.h"
#include "GB_mask_very_sparse.h"

int GB_add_sparsity         // return the sparsity structure for C
(
    // output:
    bool *apply_mask,       // if true then mask will be applied by GB_add
    // input:
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_struct, // if true, use only the structure of M
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
    // listed as sparse: it is hypersparse if M is hypersparse (and not
    // complemented), or if both A and B are hypersparse, and sparse otherwise.
    // This is determined by GB_add_phase0.  If M is complemented and all 4
    // matrices are sparse, then C=A+B is always computed.  So C is hypersparse
    // if both A and B are hypersparse, in this case.

    bool M_is_sparse_or_hyper = GB_IS_SPARSE (M) || GB_IS_HYPERSPARSE (M) ;
    bool A_is_sparse_or_hyper = GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A) ;
    bool B_is_sparse_or_hyper = GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B) ;
    bool A_is_full = GB_IS_FULL (A) ;
    bool B_is_full = GB_IS_FULL (B) ;

    if (M == NULL)
    {

        //      ------------------------------------------
        //      C       =           A       +       B
        //      ------------------------------------------
        //      sparse  .           sparse          sparse
        //      bitmap  .           sparse          bitmap
        //      full    .           sparse          full  
        //      bitmap  .           bitmap          sparse
        //      bitmap  .           bitmap          bitmap
        //      full    .           bitmap          full  
        //      full    .           full            sparse
        //      full    .           full            bitmap
        //      full    .           full            full  

        if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
        { 
            C_sparsity = GxB_SPARSE ;
        }
        else if (A_is_full || B_is_full)
        { 
            C_sparsity = GxB_FULL ;
        }
        else
        { 
            C_sparsity = GxB_BITMAP ;
        }

    }
    else if (!Mask_comp)
    {

        if (M_is_sparse_or_hyper)
        { 

            //      ------------------------------------------
            //      C      <M> =        A       +       B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse
            //      sparse  sparse      sparse          bitmap
            //      sparse  sparse      sparse          full  
            //      sparse  sparse      bitmap          sparse
            //      sparse  sparse      bitmap          bitmap
            //      sparse  sparse      bitmap          full  
            //      sparse  sparse      full            sparse
            //      sparse  sparse      full            bitmap
            //      sparse  sparse      full            full  

            // if A and B are both bitmap or full, then always use the mask.

            // if M and A and/or B are all sparse, use the mask only if
            // M is very easy to use, or if:
            // 8*nnz(M) <= ( (A sparse or hyper) ? nnz(A) : 0 ) +
            //             ( (B sparse or hyper) ? nnz(B) : 0 )

            if (A_is_sparse_or_hyper || B_is_sparse_or_hyper)
            { 
                // see Template/GB_add_sparse_M_sparse.c for a vector-by-vector
                // test of the "easy mask" condition.  This test is global for
                // all vectors of the matrices:
                bool M_is_A = GB_all_aliased (M, A) ;
                bool M_is_B = GB_all_aliased (M, B) ;
                bool all_easy_mask = Mask_struct &&
                    ((A_is_full && M_is_B) ||
                     (B_is_full && M_is_A) ||
                     (M_is_A && M_is_B)) ;

                // GB_add_sparse_template handles this case, but exploiting the
                // mask can be asympotically slow, when C and M are sparse, and
                // A and/or B are sparse, and M has many entries.  So in that
                // case, apply the mask later.
                (*apply_mask) = all_easy_mask ||
                    GB_MASK_VERY_SPARSE (8, M,
                        A_is_sparse_or_hyper ? A : NULL,
                        B_is_sparse_or_hyper ? B : NULL) ;
            }

            if (!(*apply_mask))
            {
                // redo the analysis as if there is no mask
                if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
                { 
                    C_sparsity = GxB_SPARSE ;
                }
                else if (A_is_full || B_is_full)
                { 
                    C_sparsity = GxB_FULL ;
                }
                else
                { 
                    C_sparsity = GxB_BITMAP ;
                }
            }
            else
            {
                // the mask is applied and C is sparse
                C_sparsity = GxB_SPARSE ;
            }

        }
        else
        {

            //      ------------------------------------------
            //      C      <M> =        A       +       B
            //      ------------------------------------------
            //      sparse  bitmap      sparse          sparse
            //      bitmap  bitmap      sparse          bitmap
            //      bitmap  bitmap      sparse          full  
            //      bitmap  bitmap      bitmap          sparse
            //      bitmap  bitmap      bitmap          bitmap
            //      bitmap  bitmap      bitmap          full  
            //      bitmap  bitmap      full            sparse
            //      bitmap  bitmap      full            bitmap
            //      bitmap  bitmap      full            full  

            //      ------------------------------------------
            //      C      <M> =        A       +       B
            //      ------------------------------------------
            //      sparse  full        sparse          sparse
            //      bitmap  full        sparse          bitmap
            //      bitmap  full        sparse          full  
            //      bitmap  full        bitmap          sparse
            //      bitmap  full        bitmap          bitmap
            //      bitmap  full        bitmap          full  
            //      bitmap  full        full            sparse
            //      bitmap  full        full            bitmap
            //      bitmap  full        full            full  

            // The mask is very efficient to use in the case, when C is sparse.

            if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
            { 
                C_sparsity = GxB_SPARSE ;
            }
            else
            { 
                C_sparsity = GxB_BITMAP ;
            }
        }

    }
    else // Mask_comp
    {

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      sparse  sparse      sparse          sparse      (mask later)
        //      bitmap  sparse      sparse          bitmap
        //      bitmap  sparse      sparse          full  
        //      bitmap  sparse      bitmap          sparse
        //      bitmap  sparse      bitmap          bitmap
        //      bitmap  sparse      bitmap          full  
        //      bitmap  sparse      full            sparse
        //      bitmap  sparse      full            bitmap
        //      bitmap  sparse      full            full  

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      sparse  bitmap      sparse          sparse
        //      bitmap  bitmap      sparse          bitmap
        //      bitmap  bitmap      sparse          full  
        //      bitmap  bitmap      bitmap          sparse
        //      bitmap  bitmap      bitmap          bitmap
        //      bitmap  bitmap      bitmap          full  
        //      bitmap  bitmap      full            sparse
        //      bitmap  bitmap      full            bitmap
        //      bitmap  bitmap      full            full  

        //      ------------------------------------------
        //      C     <!M> =        A       +       B
        //      ------------------------------------------
        //      sparse  full        sparse          sparse
        //      bitmap  full        sparse          bitmap
        //      bitmap  full        sparse          full  
        //      bitmap  full        bitmap          sparse
        //      bitmap  full        bitmap          bitmap
        //      bitmap  full        bitmap          full  
        //      bitmap  full        full            sparse
        //      bitmap  full        full            bitmap
        //      bitmap  full        full            full  

        if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
        { 
            // !M must be applied later if all 4 matrices are sparse or
            // hypersparse, since the GB_add_sparse_template method does not
            // handle this case.  See the "(mask later)" above.  The method can
            // construct a sparse/hyper C with !M as bitmap or full. 
            (*apply_mask) = !M_is_sparse_or_hyper ;
            C_sparsity = GxB_SPARSE ;
        }
        else
        { 
            // !M can be applied now, or later.  TODO: If M is sparse and
            // either A or B are sparse/hyper, then there might be cases where
            // !M should be applied later, for better performance.
            C_sparsity = GxB_BITMAP ;
        }
    }

    return (C_sparsity) ;
}

