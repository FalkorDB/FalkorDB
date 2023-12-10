//------------------------------------------------------------------------------
// GB_select_value_iso:  select when A is iso and the op is VALUE*
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is iso and the operator is VALUE*.

// The VALUE* operators depend only on the value of A(i,j).  Since A is iso,
// either all entries in A will be copied to C and thus C can be created as a
// shallow copy of A, or no entries from A will be copied to C and thus C is an
// empty matrix.  The select factory is not needed, except to check the iso
// value via GB_select_bitmap.

#define GB_FREE_ALL                         \
{                                           \
    GB_phybix_free (C) ;                    \
}

#include "GB_select.h"
#include "GB_scalar_wrap.h"
#include "GB_transpose.h"

GrB_Info GB_select_value_iso
(
    GrB_Matrix C,
    GrB_IndexUnaryOp op,
    GrB_Matrix A,
    int64_t ithunk,
    const GB_void *restrict athunk,
    const GB_void *restrict ythunk,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A->iso) ;
    ASSERT (op->opcode >= GB_VALUENE_idxunop_code
         && op->opcode <= GB_VALUELE_idxunop_code)
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    //--------------------------------------------------------------------------
    // determine if C is empty or a copy of A
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Type xtype = op->xtype ;
    ASSERT (xtype != NULL) ;
    const size_t xsize = xtype->size ;

    // construct a scalar S containing the iso scalar of ((xtype) A)
    struct GB_Scalar_opaque S_header ;
    GrB_Scalar S ;
    // wrap the iso-value of A in the scalar S, typecasted to xtype
    // xscalar = (op->xtype) A->x
    GB_void xscalar [GB_VLA(xsize)] ;
    GB_cast_scalar (xscalar, xtype->code, A->x, A->type->code,
        A->type->size) ;
    S = GB_Scalar_wrap (&S_header, xtype, xscalar) ;
    S->iso = false ;    // but ensure S is not iso
    ASSERT_SCALAR_OK (S, "iso scalar wrap", GB0) ;

    // apply the select operator to the iso scalar S
    GB_OK (GB_select_bitmap (C, false, op, false, (GrB_Matrix) S, ithunk,
        athunk, ythunk, Werk)) ;
    ASSERT_MATRIX_OK (C, "C from iso scalar test", GB0) ;
    bool C_empty = (GB_nnz (C) == 0) ;
    GB_phybix_free (C) ;

    //--------------------------------------------------------------------------
    // construct C: either an empty matrix, or a copy of A
    //--------------------------------------------------------------------------

    // check if C has 0 or 1 entry
    if (C_empty)
    { 
        // C is an empty matrix
        return (GB_new (&C, // existing header
            A->type, A->vlen, A->vdim, GB_Ap_calloc, true,
            GxB_AUTO_SPARSITY, GB_Global_hyper_switch_get ( ), 1)) ;
    }
    else
    { 
        // C is a shallow copy of A with all the same entries as A
        return (GB_shallow_copy (C, true, A, Werk)) ;
    }
}

