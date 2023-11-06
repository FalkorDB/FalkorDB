//------------------------------------------------------------------------------
// GxB_Vector_select: select entries from a vector (deprecated; do not use)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// DEPRECATED: use GrB_Vector_select instead.

#define GB_FREE_ALL             \
{                               \
    GrB_Scalar_free (&Thunk) ;  \
}

#include "GB_select.h"
#include "GB_get_mask.h"

GrB_Info GxB_Vector_select          // w<M> = accum (w, select(u,k))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GxB_SelectOp op_in,       // operator to apply to the entries
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar Thunk_in,      // optional input for select operator
    const GrB_Descriptor desc       // descriptor for w and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GxB_Vector_select (w, M, accum, op, u, Thunk, desc)") ;
    GB_BURBLE_START ("GxB_select:DEPRECATED") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // convert the GxB_SelectOp to a GrB_IndexUnaryOp, with a new Thunk
    //--------------------------------------------------------------------------

    GrB_IndexUnaryOp op = NULL ;
    GrB_Scalar Thunk = NULL ;
    info = GB_selectop_to_idxunop (&op, &Thunk, op_in, Thunk_in, u->type,
        Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // op is not supported, not compatible, or out of memory
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // select the entries; do not transpose; assemble pending entries
    //--------------------------------------------------------------------------

    info = GB_select (
        (GrB_Matrix) w, C_replace,          // w and its descriptor
        M, Mask_comp, Mask_struct,          // mask and its descriptor
        accum,                              // optional accum for Z=accum(C,T)
        op,                                 // operator to select the entries
        (GrB_Matrix) u,                     // first input: u
        Thunk,                              // optional input for select op
        false,                              // u, not transposed
        Werk) ;

    GB_BURBLE_END ;
    GB_FREE_ALL ;
    return (info) ;
}

