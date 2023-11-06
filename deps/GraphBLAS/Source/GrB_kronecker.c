//------------------------------------------------------------------------------
// GrB_kronecker: Kronecker product
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_kron.h"
#include "GB_get_mask.h"

//------------------------------------------------------------------------------
// GrB_Matrix_kronecker_BinaryOp: Kronecker product with binary operator
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_kronecker_BinaryOp  // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_kronecker_BinaryOp (C, M, accum, op, A, B, desc)");
    GB_BURBLE_START ("GrB_kronecker") ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_tran, B_tran, xx, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C = kron(A,B)
    //--------------------------------------------------------------------------

    // C<M> = accum (C,T) where T = kron(A,B), or with A' and/or B'
    info = GB_kron (
        C,          C_replace,      // C matrix and its descriptor
        M, Mask_comp, Mask_struct,  // mask matrix and its descriptor
        accum,                      // for accum (C,T)
        op,                         // operator that defines T=kron(A,B)
        A,          A_tran,         // A matrix and its descriptor
        B,          B_tran,         // B matrix and its descriptor
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_kronecker_Monoid: Kronecker product with monoid
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_kronecker_Monoid  // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Monoid monoid,        // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_kronecker_Monoid "
        "(C, M, accum, op, monoid, B, desc)");
    GB_BURBLE_START ("GrB_kronecker") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_tran, B_tran, xx, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C = kron(A,B)
    //--------------------------------------------------------------------------

    // C<M> = accum (C,T) where T = kron(A,B), or with A' and/or B'
    info = GB_kron (
        C,          C_replace,      // C matrix and its descriptor
        M, Mask_comp, Mask_struct,  // mask matrix and its descriptor
        accum,                      // for accum (C,T)
        monoid->op,                 // operator that defines T=kron(A,B)
        A,          A_tran,         // A matrix and its descriptor
        B,          B_tran,         // B matrix and its descriptor
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_kronecker_Semiring: Kronecker product with semiring
//------------------------------------------------------------------------------

GrB_Info GrB_Matrix_kronecker_Semiring  // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Semiring semiring,    // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_kronecker_Semiring "
        "(C, M, accum, semiring, A, B, desc)") ;
    GB_BURBLE_START ("GrB_kronecker") ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_tran, B_tran, xx, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C = kron(A,B)
    //--------------------------------------------------------------------------

    // C<M> = accum (C,T) where T = kron(A,B), or with A' and/or B'
    info = GB_kron (
        C,          C_replace,      // C matrix and its descriptor
        M, Mask_comp, Mask_struct,  // mask matrix and its descriptor
        accum,                      // for accum (C,T)
        semiring->multiply,         // operator that defines T=kron(A,B)
        A,          A_tran,         // A matrix and its descriptor
        B,          B_tran,         // B matrix and its descriptor
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_kron: Kronecker product (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_kron                   // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op,          // defines '*' for T=kron(A,B)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Matrix B,             // second input: matrix B
    const GrB_Descriptor desc       // descriptor for C, M, A, and B
)
{ 
    // call the new GrB_kronecker (the binary op version)
    return (GrB_Matrix_kronecker_BinaryOp (C, M, accum, op, A, B, desc)) ;
}

