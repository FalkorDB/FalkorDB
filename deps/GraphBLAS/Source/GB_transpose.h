//------------------------------------------------------------------------------
// GB_transpose.h:  definitions for GB_transpose
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_TRANSPOSE_H
#define GB_TRANSPOSE_H
#include "GB.h"

bool GB_transpose_method        // if true: use GB_builder, false: use bucket
(
    const GrB_Matrix A,         // matrix to transpose
    int *nworkspaces_bucket,    // # of slices of A for the bucket method
    int *nthreads_bucket        // # of threads to use for the bucket method
) ;

GrB_Info GB_transpose           // C=A', C=(ctype)A' or C=op(A')
(
    GrB_Matrix C,               // output matrix C, possibly modified in-place
    GrB_Type ctype,             // desired type of C; if NULL use A->type.
                                // ignored if op is present (cast to op->ztype)
    const bool C_is_csc,        // desired CSR/CSC format of C
    const GrB_Matrix A,         // input matrix; C == A if done in place
        // no operator is applied if op is NULL
        const GB_Operator op_in,    // unary/idxunop/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
        bool flipij,                // if true, flip i,j for user idxunop
    GB_Werk Werk
) ;

GrB_Info GB_transpose_in_place   // C=A', no change of type, no operators
(
    GrB_Matrix C,               // output matrix C, possibly modified in-place
    const bool C_is_csc,        // desired CSR/CSC format of C
    GB_Werk Werk
) ;

GrB_Info GB_transpose_cast      // C= (ctype) A' or one (A'), not in-place
(
    GrB_Matrix C,               // output matrix C, not in place
    GrB_Type ctype,             // desired type of C
    const bool C_is_csc,        // desired CSR/CSC format of C
    const GrB_Matrix A,         // input matrix; C != A
    const bool iso_one,         // if true, C = one (A'), as iso
    GB_Werk Werk
) ;

GrB_Info GB_transpose_bucket    // bucket transpose; typecast and apply op
(
    GrB_Matrix C,               // output matrix (static header)
    const GB_iso_code C_code_iso,   // iso code for C
    const GrB_Type ctype,       // type of output matrix C
    const bool C_is_csc,        // format of output matrix C
    const GrB_Matrix A,         // input matrix
        // no operator is applied if op is NULL
        const GB_Operator op,       // unary/idxunop/binop to apply
        const GrB_Scalar scalar,    // scalar to bind to binary operator
        bool binop_bind1st,         // if true, binop(x,A) else binop(A,y)
    const int nworkspaces,      // # of workspaces to use
    const int nthreads,         // # of threads to use
    GB_Werk Werk
) ;

GrB_Info GB_transpose_ix        // transpose the pattern and values of a matrix
(
    GrB_Matrix C,                       // output matrix
    const GrB_Matrix A,                 // input matrix
    // for sparse case:
    int64_t *restrict *Workspaces,      // Workspaces, size nworkspaces
    const int64_t *restrict A_slice,    // how A is sliced, size nthreads+1
    int nworkspaces,                    // # of workspaces to use
    // for all cases:
    int nthreads                        // # of threads to use
) ;

GrB_Info GB_transpose_op // transpose, typecast, and apply operator to a matrix
(
    GrB_Matrix C,                       // output matrix
    const GB_iso_code C_code_iso,       // iso code for C
        // no operator is applied if op is NULL
        const GB_Operator op,           // unary/idxunop/binop to apply
        const GrB_Scalar scalar,        // scalar to bind to binary operator
        bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    const GrB_Matrix A,                 // input matrix
    // for sparse or hypersparse case:
    int64_t *restrict *Workspaces,      // Workspaces, size nworkspaces
    const int64_t *restrict A_slice,    // how A is sliced, size nthreads+1
    int nworkspaces,                    // # of workspaces to use
    // for all cases:
    int nthreads                        // # of threads to use
) ;

GrB_Info GB_shallow_copy    // create a purely shallow matrix
(
    GrB_Matrix C,           // output matrix C, with a static header
    const bool C_is_csc,    // desired CSR/CSC format of C
    const GrB_Matrix A,     // input matrix
    GB_Werk Werk
) ;

#endif

