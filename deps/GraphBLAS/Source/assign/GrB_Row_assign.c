//------------------------------------------------------------------------------
// GrB_Row_assign: C<M'>(i,J) = accum (C(i,J),u')
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "assign/GB_assign.h"
#include "mask/GB_get_mask.h"

GrB_Info GrB_Row_assign             // C<mask'>(i,J) = accum (C(i,J),u')
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for C(i,:), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(i,J),t)
    const GrB_Vector u,             // input vector
    uint64_t i,                     // row index
    const uint64_t *J,              // column indices
    uint64_t nj,                    // number of column indices
    const GrB_Descriptor desc       // descriptor for C(i,:) and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE3 (C, mask, u,
        "GrB_Row_assign (C, M, accum, u, i, J, nj, desc)") ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_BURBLE_START ("GrB_assign") ;

    ASSERT (mask == NULL || GB_VECTOR_OK (mask)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C<M'>(i,J) = accum (C(i,J), u')
    //--------------------------------------------------------------------------

    // construct the index list I = [ i ] of length ni = 1
    uint64_t I [1] ;
    I [0] = i ;

    info = GB_assign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        true,                           // transpose the mask
        accum,                          // for accum (C(i,J),u)
        (GrB_Matrix) u, true,           // u as a matrix; always transposed
        I, false, 1,                    // a single row index
        J, false, nj,                   // column indices
        false, NULL, GB_ignore_code,    // no scalar expansion
        GB_ROW_ASSIGN,
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

