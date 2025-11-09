//------------------------------------------------------------------------------
// GB_bitmap_assign_6_whole: C bitmap, no M, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<> = A             assign
// C<> = A             subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           none
// Mask_comp:   false
// Mask_struct: true or false (ignored)
// C_replace:   false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

// If M is not present and Mask_comp is true, then an empty mask is
// complemented.  This case is handled by GB_assign_prep:  if C_replace is
// true, the matrix is cleared by GB_clear, or no action is taken otherwise.
// In either case, this method is not called.  As a result, Mask_comp and
// C_replace will always be false here.

// For scalar assignment, C = x, this method just calls GB_convert_any_to_full,
// which converts C to an iso full matrix (the iso value has already been set
// by GB_assign_prep).

// For matrix assignment, C = A, if A is sparse or hyper and C may become
// sparse or hyper, then the assignement is done by GB_subassign_24.

// If C were full: entries can be deleted if C_replace is true,
// or if A is not full and missing at least one entry.

#include "assign/GB_bitmap_assign_methods.h"
#include "assign/GB_subassign_dense.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_bitmap_assign_6_whole   // C bitmap, no M, no accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    #define C_replace false
    #define I NULL              /* I index list */
    #define I_is_32 false
    #define ni 0
    #define nI 0
    #define Ikind GB_ALL
    #define Icolon NULL
    #define J NULL              /* J index list */
    #define J_is_32 false
    #define nj 0
    #define nJ 0
    #define Jkind GB_ALL
    #define Jcolon NULL
    #define M NULL              /* mask matrix, not present here */
    #define Mask_comp false
    #define Mask_struct true
    #define accum NULL          /* not present */
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    #define assign_kind         GB_ASSIGN
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_assign_burble ("bit6_whole", C_replace, Ikind, Jkind,
        M, Mask_comp, Mask_struct, accum, A, assign_kind) ;

    GrB_Info info ;
    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT_MATRIX_OK (C, "C for bit6_whole", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bit6_whole", GB0) ;

    //--------------------------------------------------------------------------
    // C = A or C = scalar
    //--------------------------------------------------------------------------

    if (A == NULL)
    { 

        //----------------------------------------------------------------------
        // scalar assignment: C = scalar
        //----------------------------------------------------------------------

        ASSERT (C->iso) ;
        GB_convert_any_to_full (C) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C = A
        //----------------------------------------------------------------------

        if (GB_IS_BITMAP (A) || GB_IS_FULL (A))
        {

            //------------------------------------------------------------------
            // C = A where C is bitmap and A is bitmap or full
            //------------------------------------------------------------------

            // copy or typecast the values
            GB_OK (GB_cast_matrix (C, A)) ;
            int nthreads_max = GB_Context_nthreads_max ( ) ;

            if (GB_IS_BITMAP (A))
            { 
                // copy the bitmap
                GB_memcpy (C->b, A->b, GB_nnz_held (A), nthreads_max) ;
                C->nvals = GB_nnz (A) ;
            }
            else
            { 
                // free the bitmap or set it to all ones
                GB_bitmap_assign_to_full (C, nthreads_max) ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C = A where C is bitmap and A is sparse or hyper
            //------------------------------------------------------------------

            int sparsity_control =
                GB_sparsity_control (C->sparsity_control, C->vdim) ;
            if ((GB_IS_SPARSE (A) && (sparsity_control & GxB_SPARSE)) ||
                (GB_IS_HYPERSPARSE (A) && (sparsity_control & GxB_HYPERSPARSE)))
            { 
                // C becomes sparse or hypersparse, the same as A
                GB_OK (GB_subassign_24 (C, A, Werk)) ;
            }
            else
            { 
                // C remains bitmap: scatter A into the C bitmap
                GB_OK (GB_bitmap_assign_6b_whole (C, A, Werk)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "final C, bit6_whole", GB0) ;
    return (GrB_SUCCESS) ;
}

