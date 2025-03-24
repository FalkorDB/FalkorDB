//------------------------------------------------------------------------------
// GrB_Matrix_eWiseAdd: matrix element-wise operations, set union
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A+B) and variations.

#include "ewise/GB_ewise.h"
#include "mask/GB_get_mask.h"

//------------------------------------------------------------------------------
// GrB_Matrix_eWiseAdd_BinaryOp: matrix addition
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_eWiseAdd_BinaryOp       // C<M> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix Mask,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL (B) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE4 (C, Mask, A, B, "GrB_Matrix_eWiseAdd (C, M, accum, op, A, B, "
        "desc)") ;
    GB_BURBLE_START ("GrB_eWiseAdd") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_tran, B_tran, xx, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (Mask, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T) where T = A+B, A'+B, A+B', or A'+B'
    //--------------------------------------------------------------------------

    info = GB_ewise (
        C,              C_replace,  // C and its descriptor
        M, Mask_comp, Mask_struct,  // mask and its descriptor
        accum,                      // accumulate operator
        op,                         // operator that defines '+'
        A,              A_tran,     // A matrix and its descriptor
        B,              B_tran,     // B matrix and its descriptor
        true,                       // eWiseAdd
        false, NULL, NULL,          // not eWiseUnion, no alpha/beta
        Werk) ;
    GB_BURBLE_END ;

    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_eWiseAdd_Monoid: matrix addition
//------------------------------------------------------------------------------

// C<M> = accum (C,A+B) and variations.

GrB_Info GrB_Matrix_eWiseAdd_Monoid         // C<M> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Monoid monoid,        // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GrB_BinaryOp op = monoid->op ;
    return (GrB_Matrix_eWiseAdd_BinaryOp (C, M, accum, op, A, B, desc)) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_eWiseAdd_Semiring: matrix addition
//------------------------------------------------------------------------------

// C<M> = accum (C,A+B) and variations.

GrB_Info GrB_Matrix_eWiseAdd_Semiring       // C<M> = accum (C, A+B)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '+' for T=A+B
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    GrB_BinaryOp op = semiring->add->op ;
    return (GrB_Matrix_eWiseAdd_BinaryOp (C, M, accum, op, A, B, desc)) ;
}

