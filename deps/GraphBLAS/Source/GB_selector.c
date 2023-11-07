//------------------------------------------------------------------------------
// GB_selector:  select entries from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_selector does the work for GB_select.  It also deletes zombies for
// GB_wait using the GxB_NONZOMBIE operator, deletes entries outside a smaller
// matrix for GxB_*resize using GrB_ROWLE, and extracts the diagonal entries
// for GB_Vector_diag.

// For GB_resize (using GrB_ROWLE) and GB_wait (using GxB_NONZOMBIE), C may be
// NULL.  In this case, A is always sparse or hypersparse.  If C is NULL on
// input, A is modified in-place.  Otherwise, C is an uninitialized static
// header.

// TODO: GB_selector does not exploit the mask.

#include "GB_select.h"

#define GB_FREE_ALL ;

GrB_Info GB_selector
(
    GrB_Matrix C,               // output matrix, NULL or existing header
    const GrB_IndexUnaryOp op,
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const GrB_Scalar Thunk,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for GB_selector", GB0) ;
    ASSERT_SCALAR_OK (Thunk, "Thunk for GB_selector", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_selector", GB_FLIP (GB0)) ;
    // positional op (tril, triu, diag, offdiag, resize, rowindex, ...):
    // can't be jumbled.  nonzombie, entry-valued op, user op: jumbled OK
    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IMPLIES (GB_OPCODE_IS_POSITIONAL (opcode), !GB_JUMBLED (A))) ;
    ASSERT (C == NULL || (C != NULL && (C->static_header || GBNSTATIC))) ;

    bool in_place_A = (C == NULL) ; // GrB_wait and GB_resize only
    const bool A_iso = A->iso ;

    //--------------------------------------------------------------------------
    // get Thunk
    //--------------------------------------------------------------------------

    // get the type of the thunk input of the operator
    ASSERT (GB_nnz ((GrB_Matrix) Thunk) > 0) ;
    const GB_Type_code tcode = Thunk->type->code ;

    // ythunk = (op->ytype) Thunk
    size_t ysize = op->ytype->size ;
    GB_void ythunk [GB_VLA(ysize)] ;
    memset (ythunk, 0, ysize) ;
    GB_cast_scalar (ythunk, op->ytype->code, Thunk->x, tcode, ysize) ;

    // ithunk = (int64) Thunk, if compatible
    int64_t ithunk = 0 ;
    if (GB_Type_compatible (GrB_INT64, Thunk->type))
    {
        GB_cast_scalar (&ithunk, GB_INT64_code, Thunk->x, tcode,
            sizeof (int64_t)) ;
    }

    // athunk = (A->type) Thunk, for VALUEEQ operator only
    const size_t asize = A->type->size ;
    GB_void athunk [GB_VLA(asize)] ;
    memset (athunk, 0, asize) ;
    if (opcode == GB_VALUEEQ_idxunop_code)
    {
        ASSERT (GB_Type_compatible (A->type, Thunk->type)) ;
        GB_cast_scalar (athunk, A->type->code, Thunk->x, tcode, asize) ;
    }

    //--------------------------------------------------------------------------
    // determine if C is iso for a non-iso A
    //--------------------------------------------------------------------------

    bool C_iso = A_iso ||                       // C iso value is Ax [0]
        (opcode == GB_VALUEEQ_idxunop_code) ;   // C iso value is thunk
    if (C_iso)
    { 
        GB_BURBLE_MATRIX (A, "(iso select) ") ;
    }

    //--------------------------------------------------------------------------
    // handle iso case for built-in ops that depend only on the value
    //--------------------------------------------------------------------------

    if (A_iso && opcode >= GB_VALUENE_idxunop_code
              && opcode <= GB_VALUELE_idxunop_code)
    { 
        return (GB_select_value_iso (C, op, A, ithunk, athunk, ythunk, Werk)) ;
    }

    // The CUDA select kernel would be called here.

    //--------------------------------------------------------------------------
    // bitmap/as-if-full case
    //--------------------------------------------------------------------------

    bool use_select_bitmap ;
    if (opcode == GB_NONZOMBIE_idxunop_code || in_place_A)
    { 
        // GB_select_bitmap does not support the nonzombie opcode, nor does
        // it support operating on A in place.  For the NONZOMBIE operator, A
        // will never be bitmap.
        use_select_bitmap = false ;
    }
    else if (opcode == GB_DIAG_idxunop_code)
    { 
        // GB_select_bitmap supports the DIAG operator, but it is currently
        // not efficient (GB_select_bitmap should return a sparse diagonal
        // matrix, not bitmap).  So use the sparse case if A is not bitmap,
        // since the sparse case below does not support the bitmap case.
        use_select_bitmap = GB_IS_BITMAP (A) ;
    }
    else
    { 
        // For bitmap, full, or as-if-full matrices (sparse/hypersparse with
        // all entries present, not jumbled, no zombies, and no pending
        // tuples), use the bitmap selector for all other operators (TRIL,
        // TRIU, OFFDIAG, NONZERO, EQ*, GT*, GE*, LT*, LE*, and user-defined
        // operators).
        use_select_bitmap = GB_IS_BITMAP (A) || GB_IS_FULL (A) ;
    }

    if (use_select_bitmap)
    { 
        GB_BURBLE_MATRIX (A, "(bitmap select) ") ;
        ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
        return (GB_select_bitmap (C, C_iso, op,                  
            flipij, A, ithunk, athunk, ythunk, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // column selector
    //--------------------------------------------------------------------------

    if (opcode == GB_COLINDEX_idxunop_code ||
        opcode == GB_COLLE_idxunop_code ||
        opcode == GB_COLGT_idxunop_code)
    { 
        return (GB_select_column (C, C_iso, op, A, ithunk, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // sparse/hypersparse general case
    //--------------------------------------------------------------------------

    return (GB_select_sparse (C, C_iso, op, flipij, A, ithunk, athunk, ythunk,
        Werk)) ;
}

