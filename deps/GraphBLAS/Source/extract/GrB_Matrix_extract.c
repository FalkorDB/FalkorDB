//------------------------------------------------------------------------------
// GrB_Matrix_extract: C<M> = accum (C, A(I,J)) or A(J,I)'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "extract/GB_extract.h"
#include "mask/GB_get_mask.h"

GrB_Info GrB_Matrix_extract         // C<M> = accum (C, A(I,J))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // first input:  matrix A
    const uint64_t *I,              // row indices (64-bit)
    uint64_t ni,                    // number of row indices
    const uint64_t *J,              // column indices (64-bit)
    uint64_t nj,                    // number of column indices
    const GrB_Descriptor desc       // descriptor for C, M, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE3 (C, Mask, A,
        "GrB_Matrix_extract (C, M, accum, A, I, ni, J, nj, desc)") ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_BURBLE_START ("GrB_extract") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (Mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // do the work in GB_extract
    //--------------------------------------------------------------------------

    info = GB_extract (
        C,      C_replace,          // output matrix C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // optional accum for Z=accum(C,T)
        A,      A_transpose,        // A and its descriptor
        I, false, ni,               // row indices (64-bit)
        J, false, nj,               // column indices (64-bit)
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

