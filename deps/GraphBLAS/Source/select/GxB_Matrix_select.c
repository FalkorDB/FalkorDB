//------------------------------------------------------------------------------
// GxB_Matrix_select: select entries from a matrix: deprecated; do not use
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// DEPRECATED: use GrB_Matrix_select instead.

#define GB_FREE_ALL             \
{                               \
    GrB_Scalar_free (&Thunk) ;  \
}

#include "select/GB_select.h"
#include "mask/GB_get_mask.h"

GrB_Info GxB_Matrix_select  // C<M> = accum (C, select(A,k)) or select(A',k)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GxB_SelectOp op_in,       // operator to select the entries
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Scalar Thunk_in,      // optional input for select operator
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE4 (C, M_in, A, Thunk_in,
        "GxB_Matrix_select (C, M, accum, op, A, Thunk, desc)") ;
    GB_BURBLE_START ("GxB_select:DEPRECATED") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // convert the GxB_SelectOp to a GrB_IndexUnaryOp, with a new Thunk
    //--------------------------------------------------------------------------

    GrB_IndexUnaryOp op = NULL ;
    GrB_Scalar Thunk = NULL ;
    info = GB_selectop_to_idxunop (&op, &Thunk, op_in, Thunk_in, A->type,
        Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // op is not supported, not compatible, or out of memory
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // select the entries and optionally transpose; assemble pending tuples
    //--------------------------------------------------------------------------

    info = GB_select (
        C, C_replace,               // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(C,T)
        op,                         // operator to select the entries
        A,                          // first input: A
        Thunk,                      // optional input for select operator
        A_transpose,                // descriptor for A
        Werk) ;

    GB_BURBLE_END ;
    GB_FREE_ALL ;
    return (info) ;
}

