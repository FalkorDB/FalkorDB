//------------------------------------------------------------------------------
// GrB_transpose: transpose a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A') or accum (C,A)

#define GB_FREE_ALL GB_Matrix_free (&T) ;

#include "GB_transpose.h"
#include "GB_accum_mask.h"
#include "GB_get_mask.h"

GrB_Info GrB_transpose              // C<M> = accum(C,A') or accum(C,A)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    // C may be aliased with M and/or A

    GB_WHERE (C, "GrB_transpose (C, M, accum, A, desc)") ;
    GB_BURBLE_START ("GrB_transpose") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    ASSERT_MATRIX_OK (C, "C input for GrB_transpose", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M_in, "M for GrB_transpose", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GrB_transpose", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GrB_transpose", GB0) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, A->type, Werk));

    // check the dimensions
    int64_t tnrows = (!A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t tncols = (!A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    if (GB_NROWS (C) != tnrows || GB_NCOLS (C) != tncols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Dimensions not compatible:\n"
            "output is " GBd "-by-" GBd "\n"
            "input is " GBd "-by-" GBd "%s",
            GB_NROWS (C), GB_NCOLS (C),
            tnrows, tncols, (!A_transpose) ? " (transposed)" : "") ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    //--------------------------------------------------------------------------
    // T = A or A', where T can have the type of C or the type of A
    //--------------------------------------------------------------------------

    GB_CLEAR_STATIC_HEADER (T, &T_header) ;
    bool C_is_csc = C->is_csc ;
    if (C_is_csc != A->is_csc)
    { 
        // Flip the sense of A_transpose
        A_transpose = !A_transpose ;
    }

    if (!A_transpose)
    {

        // T = A', the default behavior.  This step may seem counter-intuitive,
        // but method computes C<M>=A' by default when A_transpose is false.

        // Precasting:
        if (accum == NULL)
        { 
            // If there is no accum operator, T is transplanted into Z and
            // typecasted into the C->type during the transpose.
            GB_OK (GB_transpose_cast (T, C->type, C_is_csc, A, false,
                Werk)) ;
        }
        else
        { 
            // If the accum operator is present, entries in the intersection of
            // T and C are typecasted into the accum->ytype, while entries in T
            // but not C are typecasted directly into C->type.  Thus, the
            // typecast of T (if any) must wait, and be done in call to GB_add
            // in GB_accum_mask.
            GB_OK (GB_transpose_cast (T, A->type, C_is_csc, A, false,
                Werk)) ;
        }

        // no operator; typecasting done if accum is NULL
    }
    else
    { 

        // T = A, a pure shallow copy; nothing at all is allocated.  No
        // typecasting is done since the types of T and A are the same.  If the
        // A_transpose descriptor is true, A is viewed as transposed first.
        // The method transposes A again, giving T=A''=A.  This is a double
        // transpose, so C<M>=A is computed, and no transpose is done.  T is
        // typecasted eventually, into the type of C if the types of T and C
        // differ.  That can be postponed at no cost since the following step
        // is free.
        GBURBLE ("(cheap) ") ;
        // set T->iso = A->iso  OK
        GB_OK (GB_shallow_copy (T, C_is_csc, A, Werk)) ;
    }

    ASSERT (T->is_csc == C->is_csc) ;
    ASSERT_MATRIX_OK (T, "T for GrB_transpose", GB0) ;
    ASSERT_MATRIX_OK (C, "C for GrB_transpose", GB0) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask M
    //--------------------------------------------------------------------------

    info = GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp, 
        Mask_struct, Werk) ;
    if (info == GrB_SUCCESS)
    {
        ASSERT_MATRIX_OK (C, "final C for GrB_transpose", GB0) ;
    }

    GB_BURBLE_END ;
    GB_FREE_ALL ;
    return (info) ;
}

