//------------------------------------------------------------------------------
// GB_bitmap_assign: assign to C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Implements GrB_Row_assign, GrB_Col_assign, GrB_assign, GxB_subassign when C
// is in bitmap form, or when C is converted into bitmap form.

// C is returned as bitmap in all cases except for C = A or C = scalar (the
// whole_C_matrix case with GB_bitmap_assign_6_whole).  For that
// method, C can be returned with any sparsity structure.

#include "assign/GB_bitmap_assign_methods.h"
#define GB_GENERIC
#include "assign/include/GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_phybix_free (C) ;

GrB_Info GB_bitmap_assign
(
    // input/output:
    GrB_Matrix C,               // input/output matrix
    // inputs:
    const bool C_replace,       // descriptor for C
    const GrB_Index *I,         // I index list
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,         // J index list
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,         // mask matrix, NULL if not present
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // NULL if not present
    const GrB_Matrix A,         // input matrix, NULL for scalar assignment
    const void *scalar,         // input scalar, if A == NULL
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign", GB0) ;

    //--------------------------------------------------------------------------
    // ensure C is in bitmap form
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_any_to_bitmap (C, Werk)) ;
    ASSERT (GB_IS_BITMAP (C)) ;

    bool whole_C_matrix = (Ikind == GB_ALL && Jkind == GB_ALL) ;

    //--------------------------------------------------------------------------
    // do the assignment
    //--------------------------------------------------------------------------

    if (M == NULL)
    {
        if (accum == NULL)
        {
            if (whole_C_matrix)
            { 
                // C = A or scalar, no mask.  C may become sparse, hyper, or
                // full, or it may remain bitmap.  The Mask_comp = true and/or
                // C_replace = true cases are handled in GB_assign_prep, and
                // in that case, GB_bitmap_assign is not called.
                ASSERT (!C_replace) ;
                ASSERT (!Mask_comp) ;
                GB_OK (GB_bitmap_assign_6_whole (C,
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C(I,J) = A or scalar, no mask
                GB_OK (GB_bitmap_assign_6 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    /* no M, */ Mask_comp, Mask_struct, /* no accum, */
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
        else
        {
            if (whole_C_matrix)
            { 
                // C += A or scalar, no mask.
                GB_OK (GB_bitmap_assign_5_whole (C, C_replace,
                    /* no M, */ Mask_comp, Mask_struct, accum,
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C(I,J) += A or scalar, no mask.
                GB_OK (GB_bitmap_assign_5 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    /* no M, */ Mask_comp, Mask_struct, accum,
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
    }
    else if (GB_IS_BITMAP (M) || GB_IS_FULL (M))
    {
        if (accum == NULL)
        {
            if (whole_C_matrix)
            { 
                // C<M or !M, where M is bitmap or full> = A or scalar
                GB_OK (GB_bitmap_assign_2_whole (C, C_replace,
                    M, Mask_comp, Mask_struct, /* no accum, */
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C<M or !M, where M is bitmap or full>(I,J) = A or scalar
                GB_OK (GB_bitmap_assign_2 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    M, Mask_comp, Mask_struct, /* no accum, */
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
        else
        {
            if (whole_C_matrix)
            { 
                // C<M or !M, where M is bitmap or full> += A or scalar
                GB_OK (GB_bitmap_assign_1_whole (C, C_replace,
                    M, Mask_comp, Mask_struct, accum,
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C<M or !M, where M is bitmap or full>(I,J) + A or scalar
                GB_OK (GB_bitmap_assign_1 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    M, Mask_comp, Mask_struct, accum,
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
    }
    else if (!Mask_comp)
    {
        if (accum == NULL)
        {
            if (whole_C_matrix)
            { 
                // C<M> = A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_4_whole (C, C_replace,
                    M, /* Mask_comp false, */ Mask_struct, /* no accum, */
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C<M>(I,J) = A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_4 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    M, /* Mask_comp false, */ Mask_struct, /* no accum, */
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
        else
        {
            if (whole_C_matrix)
            { 
                // C<M> += A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_3_whole (C, C_replace,
                    M, /* Mask_comp false, */ Mask_struct, accum,
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C<M>(I,J) += A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_3 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    M, /* Mask_comp false, */ Mask_struct, accum,
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
    }
    else // Mask_comp is true
    {
        if (accum == NULL)
        {
            if (whole_C_matrix)
            { 
                // C<!M> = A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_8_whole (C, C_replace,
                    M, /* Mask_comp true, */ Mask_struct, /* no accum, */
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C<!M>(I,J) = A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_8 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    M, /* Mask_comp true, */ Mask_struct, /* no accum, */
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
        else
        {
            if (whole_C_matrix)
            { 
                // C<!M> += A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_7_whole (C, C_replace,
                    M, /* Mask_comp true, */ Mask_struct, accum,
                    A, scalar, scalar_type, Werk)) ;
            }
            else
            { 
                // C<!M>(I,J) += A or scalar, M is sparse or hypersparse
                GB_OK (GB_bitmap_assign_7 (C, C_replace,
                    I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
                    M, /* Mask_comp true, */ Mask_struct, accum,
                    A, scalar, scalar_type, assign_kind, Werk)) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "final C for bitmap assign", GB0) ;
    return (GrB_SUCCESS) ;
}

