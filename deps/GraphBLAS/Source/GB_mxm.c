//------------------------------------------------------------------------------
// GB_mxm: matrix-matrix multiply for GrB_mxm, GrB_mxv, and GrB_vxm
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A*B) and variations.

// This function is not user-callable.  It does the work for user-callable
// functions GrB_mxm, GrB_mxv, and GrB_vxm.

#define GB_FREE_ALL         \
{                           \
    GB_Matrix_free (&MT) ;  \
    GB_Matrix_free (&T) ;   \
}

#include "GB_mxm.h"
#include "GB_accum_mask.h"

GrB_Info GB_mxm                     // C<M> = A*B
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M_input,       // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' and '*' for C=A*B
    const GrB_Matrix A,             // input matrix
    const bool A_transpose,         // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    const bool B_transpose,         // if true, use B' instead of B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    const GrB_Desc_Value AxB_method,// for auto vs user selection of methods
    const int do_sort,              // if nonzero, try to return C unjumbled
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M, A, and/or B

    GrB_Info info ;

    struct GB_Matrix_opaque MT_header, T_header ;
    GrB_Matrix MT = NULL, T = NULL ;

    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;

    ASSERT_MATRIX_OK (C, "C input for GB_mxm", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M_input, "M for GB_mxm", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_mxm", GB0) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for GB_mxm", GB0) ;
    ASSERT_MATRIX_OK (A, "A for GB_mxm", GB0) ;
    ASSERT_MATRIX_OK (B, "B for GB_mxm", GB0) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Type T_type = semiring->add->op->ztype ;
    GB_OK (GB_compatible (C->type, C, M_input, Mask_struct, accum, T_type,
        Werk)) ;

    // T=A*B via semiring: A and B must be compatible with semiring->multiply
    if (flipxy)
    { 
        // z=fmult(b,a), for entries a from A, and b from B
        GB_OK (GB_BinaryOp_compatible (semiring->multiply,
                NULL, B->type, A->type, GB_ignore_code, Werk)) ;
    }
    else
    { 
        // z=fmult(a,b), for entries a from A, and b from B
        GB_OK (GB_BinaryOp_compatible (semiring->multiply,
                NULL, A->type, B->type, GB_ignore_code, Werk)) ;
    }

    // check the dimensions
    int64_t anrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t ancols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    int64_t bnrows = (B_transpose) ? GB_NCOLS (B) : GB_NROWS (B) ;
    int64_t bncols = (B_transpose) ? GB_NROWS (B) : GB_NCOLS (B) ;
    if (ancols != bnrows || GB_NROWS (C) != anrows || GB_NCOLS (C) != bncols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Dimensions not compatible:\n"
            "output is " GBd "-by-" GBd "\n"
            "first input is " GBd "-by-" GBd "%s\n"
            "second input is " GBd "-by-" GBd "%s",
            GB_NROWS (C), GB_NCOLS (C),
            anrows, ancols, A_transpose ? " (transposed)" : "",
            bnrows, bncols, B_transpose ? " (transposed)" : "") ;
    }

    //--------------------------------------------------------------------------
    // finish any pending work and check for C<!NULL> mask
    //--------------------------------------------------------------------------

    GrB_Matrix M = M_input ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (M) ;
    if (Mask_struct && GB_as_if_full (M))
    { 
        // ignore the mask if all entries present and not complemented
        M = NULL ;
    }

    // quick return if a NULL mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (B) ;

    //--------------------------------------------------------------------------
    // T = A*B, A'*B, A*B', or A'*B', also using the mask if present
    //--------------------------------------------------------------------------

    // If C is dense (with no pending work), and the accum is present, then
    // C+=A*B can be done in-place (C_replace is effectively false).  If C is
    // dense, M is present, and C_replace is false, then C<M>+=A*B or
    // C<!M>+=A*B can also be done in-place.  In all of these cases, C remains
    // dense with all entries present.  C can have any sparsity structure;
    // its pattern is ignored.

    // If C is bitmap, then it can always be be done in-place (assuming the
    // type of C is OK).  The accum operator need not be present.  GB_AxB_meta
    // can easily insert non-entries into C and check for non-entries, via the
    // bitmap.

    // To compute C in-place, its type must match the accum->ztype, or the
    // semiring->add->ztype if accum is not present.  To compute in-place,
    // C must also not be transposed, and it cannot be aliased with M, A, or B.

    GB_CLEAR_STATIC_HEADER (MT, &MT_header) ;
    GB_CLEAR_STATIC_HEADER (T, &T_header) ;

    bool mask_applied = false ;
    bool done_in_place = false ;
    bool M_transposed = false ;
    GB_OK (GB_AxB_meta (T, C, C_replace, C->is_csc, MT, &M_transposed, M,
        Mask_comp, Mask_struct, accum, A, B, semiring, A_transpose,
        B_transpose, flipxy, &mask_applied, &done_in_place, AxB_method,
        do_sort, Werk)) ;

    if (done_in_place)
    { 
        // C has been computed in-place; no more work to do
        GB_FREE_ALL ;
        GB_OK (GB_conform (C, Werk)) ;
        ASSERT_MATRIX_OK (C, "C from GB_mxm (in-place)", GB0) ;
        return (info) ;
    }

    ASSERT_MATRIX_OK (T, "T=A*B from GB_AxB_meta", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M_transposed ? MT : NULL, "MT from meta", GB0) ;
    ASSERT (GB_ZOMBIES_OK (T)) ;
    ASSERT (GB_JUMBLED_OK (T)) ;
    ASSERT (!GB_PENDING (T)) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask
    //--------------------------------------------------------------------------

    if ((accum == NULL) && (C->is_csc == T->is_csc)
        && (M == NULL || (M != NULL && mask_applied))
        && (C_replace || GB_NNZ_UPPER_BOUND (C) == 0))
    {
        // C = 0 ; C = (ctype) T ; with the same CSR/CSC format.  The mask M
        // (if any) has already been applied.  If C is also empty, or to be
        // cleared anyway, and if accum is not present, then T can be
        // transplanted directly into C, as C = (ctype) T, typecasting if
        // needed.  If no typecasting is done then this takes no time at all
        // and is a pure transplant.  Also conform C to its desired
        // hypersparsity.
        GB_Matrix_free (&MT) ;
        if (GB_ZOMBIES (T) && T->type != C->type)
        { 
            // T = A*B can be constructed with zombies, using the dot3 method.
            // Since its type differs from C, its values will be typecasted
            // from T->type to C->type.  The zombies are killed before
            // typecasting.  Otherwise, if they were not killed, uninitialized
            // values in T->x for these zombies will get typecasted into C->x.
            // Typecasting a zombie is safe, since the values of all zombies
            // are ignored.  But valgrind complains about it, so they are
            // killed now.  Also see the discussion in GB_transplant.
            GBURBLE ("(wait, so zombies are not typecasted) ") ;
            GB_OK (GB_wait (T, "T", Werk)) ;
        }
        GB_OK (GB_transplant_conform (C, C->type, &T, Werk)) ;
        // C may be returned with zombies and jumbled, but no pending tuples
        ASSERT_MATRIX_OK (C, "C from GB_mxm (transplanted)", GB0) ;
        ASSERT (GB_ZOMBIES_OK (C)) ;
        ASSERT (GB_JUMBLED_OK (C)) ;
        ASSERT (!GB_PENDING (C)) ;
        info = GB_block (C, Werk) ;

    }
    else
    { 
        // C<M> = accum (C,T)
        // GB_accum_mask also conforms C to its desired hypersparsity.
        info = GB_accum_mask (C, M, (M_transposed) ? MT : NULL, accum, &T,
            C_replace, Mask_comp, Mask_struct, Werk) ;
        GB_Matrix_free (&MT) ;
        #ifdef GB_DEBUG
        if (info == GrB_SUCCESS)
        {
            // C may be returned jumbled, with zombies and pending tuples
            ASSERT_MATRIX_OK (C, "Final C from GB_mxm (accum_mask)", GB0) ;
            ASSERT (GB_ZOMBIES_OK (C)) ;
            ASSERT (GB_JUMBLED_OK (C)) ;
            ASSERT (GB_PENDING_OK (C)) ;
        }
        #endif
    }

    return (info) ;
}

