//------------------------------------------------------------------------------
// GrB_Matrix_assign: C<M>(I,J) = accum (C(I,J),A) or A'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "assign/GB_assign.h"
#include "mask/GB_get_mask.h"

GrB_Info GrB_Matrix_assign          // C<M>(I,J) += A or A'
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // mask for C, unused if NULL
    const GrB_BinaryOp accum,       // accum for Z=accum(C(I,J),T)
    const GrB_Matrix A,             // first input:  matrix A
    const uint64_t *I,              // row indices
    uint64_t ni,                    // number of row indices
    const uint64_t *J,              // column indices
    uint64_t nj,                    // number of column indices
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE3 (C, Mask, A,
        "GrB_Matrix_assign (C, M, accum, A, I, ni, J, nj, desc)") ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_BURBLE_START ("GrB_assign") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (Mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C<M>(I,J) = accum (C(I,J), A)
    //--------------------------------------------------------------------------

    info = GB_assign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask matrix and its descriptor
        false,                          // do not transpose the mask
        accum,                          // for accum (C(I,J),A)
        A, A_transpose,                 // A and its descriptor (T=A or A')
        I, false, ni,                   // row indices
        J, false, nj,                   // column indices
        false, NULL, GB_ignore_code,    // no scalar expansion
        GB_ASSIGN,
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

