//------------------------------------------------------------------------------
// GB_apply.h: definitions for GB_apply
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_APPLY_H
#define GB_APPLY_H
#include "GB.h"
#include "GB_unop.h"

GrB_Info GB_apply                   // C<M> = accum (C, op(A)) or op(A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // M descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
        const GB_Operator op_in,        // unary/idxunop/binop to apply
        const GrB_Scalar scalar_in,     // scalar to bind to binop, or thunk
        bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    const GrB_Matrix A,             // first or 2nd input:  matrix A
    bool A_transpose,               // A matrix descriptor
    GB_Werk Werk
) ;

// Cx and A->x may be aliased in GB_apply_op

GrB_Info GB_apply_op        // apply a unary op, idxunop, or binop, Cx = op (A)
(
    GB_void *Cx,                    // output array
    const GrB_Type ctype,           // type of C
    const GB_iso_code C_code_iso,   // C non-iso, or code to compute C iso value
        const GB_Operator op,       // unary/index-unary/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, C=binop(s,A), else C=binop(A,s)
        bool flipij,                // if true, flip i,j for user idxunop
    const GrB_Matrix A,             // input matrix
    GB_Werk Werk
) ;

GrB_Info GB_shallow_op      // create shallow matrix and apply operator
(
    GrB_Matrix C,           // output C, of type op*->ztype, static header
    const bool C_is_csc,    // desired CSR/CSC format of C
        const GB_Operator op,       // unary/index-unary/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
        bool flipij,                // if true, flip i,j for user idxunop
    const GrB_Matrix A,     // input matrix to typecast
    GB_Werk Werk
) ;

#endif
