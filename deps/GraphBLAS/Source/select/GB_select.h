//------------------------------------------------------------------------------
// GB_select.h: definitions for GrB_select and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SELECT_H
#define GB_SELECT_H
#include "GB.h"
#include "math/GB_math.h"
#include "select/GB_select_iso.h"

GrB_Info GB_select          // C<M> = accum (C, select(A,k)) or select(A',k)
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // descriptor for M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_IndexUnaryOp op_in,
    const GrB_Matrix A,             // input matrix
    const GrB_Scalar Thunk,         // always present
    const bool A_transpose,         // A matrix descriptor
    GB_Werk Werk
) ;

GrB_Info GB_selector
(
    GrB_Matrix C,               // output matrix, NULL or existing header
    const GrB_IndexUnaryOp op,
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const GrB_Scalar Thunk,
    GB_Werk Werk
) ;

GrB_Info GB_select_sparse
(
    GrB_Matrix C,                   // output matrix; empty header on input
    const bool C_iso,               // if true, construct C as iso
    const GrB_IndexUnaryOp op,
    const bool flipij,              // if true, flip i and j for the op
    const GrB_Matrix A,             // input matrix
    const int64_t ithunk,           // input scalar, cast to int64_t
    const GB_void *restrict athunk, // same input scalar, but cast to A->type
    const GB_void *restrict ythunk, // same input scalar, but cast to op->ytype
    GB_Werk Werk
) ;

GrB_Info GB_select_value_iso
(
    GrB_Matrix C,
    GrB_IndexUnaryOp op,
    GrB_Matrix A,
    int64_t ithunk,
    const GB_void *restrict athunk,
    const GB_void *restrict ythunk,
    GB_Werk Werk
) ;

GrB_Info GB_select_column
(
    GrB_Matrix C,
    const GrB_IndexUnaryOp op,
    GrB_Matrix A,
    int64_t ithunk,
    GB_Werk Werk
) ;

GrB_Info GB_select_bitmap
(
    GrB_Matrix C,               // output matrix, static header
    const bool C_iso,           // if true, C is iso
    const GrB_IndexUnaryOp op,
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const int64_t ithunk,       // (int64_t) Thunk, if Thunk is NULL
    const GB_void *restrict athunk,     // (A->type) Thunk
    const GB_void *restrict ythunk,     // (op->ytype) Thunk
    GB_Werk Werk
) ;

GrB_Info GB_selectop_to_idxunop
(
    // output:
    GrB_IndexUnaryOp *idxunop_handle,
    GrB_Scalar *NewThunk_handle,
    // input:
    GxB_SelectOp selectop,
    GrB_Scalar Thunk,
    GrB_Type atype,
    GB_Werk Werk
) ;

GrB_Info GB_select_generic_phase1
(
    // input/output:
    GrB_Matrix C,
    uint64_t *restrict Wfirst,
    uint64_t *restrict Wlast,
    // input:
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_generic_phase2
(
    // input/output:
    GrB_Matrix C,
    // input:
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_positional_phase1
(
    // input/output:
    GrB_Matrix C,
    // output:
    void *Zp,                       // if C->p_is_32: 32 bit, else 64-bit
    uint64_t *restrict Wfirst,
    uint64_t *restrict Wlast,
    // input:
    const GrB_Matrix A,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_positional_phase2
(
    // input/output:
    GrB_Matrix C,
    // input:
    const void *Zp,                 // if C->p_is_32: 32 bit, else 64-bit
    const uint64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_positional_bitmap
(
    // input/output:
    GrB_Matrix C,                   // C->b and C->nvals are computed
    // input:
    GrB_Matrix A,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

GrB_Info GB_select_generic_bitmap
(
    // input/output:
    GrB_Matrix C,                   // C->b and C->nvals are computed
    // input:
    GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

#endif

