//------------------------------------------------------------------------------
// GB_add.h: definitions for GB_add and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ADD_H
#define GB_ADD_H
#include "GB.h"
#include "math/GB_math.h"

GrB_Info GB_add             // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B,     // input B matrix
    const bool is_eWiseUnion,   // if true, eWiseUnion, else eWiseAdd
    const GrB_Scalar alpha, // alpha and beta ignored for eWiseAdd,
    const GrB_Scalar beta,  // nonempty scalars for GxB_eWiseUnion
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    const bool flipij,      // if true, i,j must be flipped
    const bool A_and_B_are_disjoint,   // if true, A and B are disjoint
    GB_Werk Werk
) ;

GrB_Info GB_add_phase1                  // count nnz in each C(:,j)
(
    // output of phase1:
    void **Cp_handle,                   // output of size Cnvec+1
    size_t *Cp_size_handle,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    const bool A_and_B_are_disjoint,    // if true, then A and B are disjoint
    // tasks from phase0b:
    GB_task_struct *restrict TaskList,  // array of structs
    const int C_ntasks,                 // # of tasks
    const int C_nthreads,               // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    const void *Ch,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,        // if true, then Ch == M->h
    const bool Cp_is_32,        // if true, Cp is 32-bit; else 64-bit
    const bool Cj_is_32,        // if true, Ch is 32-bit; else 64-bit
    // original input:
    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_struct,     // if true, use the only structure of M
    const bool Mask_comp,       // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Werk Werk
) ;

GrB_Info GB_add_phase2      // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix C,           // output matrix, static header
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    const bool flipij,      // if true, i,j must be flipped
    const bool A_and_B_are_disjoint,    // if true, then A and B are disjoint
    // from phase1:
    void **Cp_handle,       // vector pointers for C
    size_t Cp_size,
    const int64_t Cnvec_nonempty,   // # of non-empty vectors in C
    // tasks from phase1a:
    const GB_task_struct *restrict TaskList,    // array of structs
    const int C_ntasks,         // # of tasks
    const int C_nthreads,       // # of threads to use
    // analysis from phase0:
    const int64_t Cnvec,
    void **Ch_handle,
    size_t Ch_size,
    const int64_t *restrict C_to_M,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,        // if true, then Ch == M->h
    const bool Cp_is_32,        // if true, Cp is 32-bit; else 64-bit
    const bool Cj_is_32,        // if true, Ch is 32-bit; else 64-bit
    const bool Ci_is_32,        // if true, Ci is 32-bit; else 64-bit
    const int C_sparsity,
    // original input:
    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_struct,     // if true, use the only structure of M
    const bool Mask_comp,       // if true, use !M
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool is_eWiseUnion,   // if true, eWiseUnion, else eWiseAdd
    const GrB_Scalar alpha,     // alpha and beta ignored for eWiseAdd,
    const GrB_Scalar beta,      // nonempty scalars for GxB_eWiseUnion
    GB_Werk Werk
) ;

int GB_add_sparsity         // return the sparsity structure for C
(
    // output:
    bool *apply_mask,       // if true then mask will be applied by GB_add
    // input:
    const GrB_Matrix M,     // optional mask for C, unused if NULL
    const bool Mask_struct, // if true, use only the structure of M
    const bool Mask_comp,   // if true, use !M
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B      // input B matrix
) ;

bool GB_add_iso             // c = op(a,b), return true if C is iso
(
    // output
    GB_void *restrict c,    // output scalar of iso array
    // input
    GrB_Type ctype,         // type of c
    GrB_Matrix A,           // input matrix
    const GB_void *restrict alpha_scalar,   // of type op->xtype
    GrB_Matrix B,           // input matrix
    const GB_void *restrict beta_scalar,    // of type op->ytype
    GrB_BinaryOp op,        // binary operator
    const bool A_and_B_are_disjoint,
    const bool is_eWiseUnion
) ;

#endif

