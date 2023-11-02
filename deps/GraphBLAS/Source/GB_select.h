//------------------------------------------------------------------------------
// GB_select.h: definitions for GrB_select and related functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SELECT_H
#define GB_SELECT_H
#include "GB.h"
#include "GB_is_nonzero.h"

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
    GrB_Matrix C,
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const int64_t ithunk,
    const GB_void *restrict athunk,
    const GB_void *restrict ythunk,
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
    const bool C_iso,
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
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
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
    int64_t *restrict Ci,
    GB_void *restrict Cx,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
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
    int64_t *restrict Zp,
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

GrB_Info GB_select_positional_phase2
(
    int64_t *restrict Ci,
    GB_void *restrict Cx,
    const int64_t *restrict Zp,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
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
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const int64_t ithunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

GrB_Info GB_select_generic_bitmap
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

//------------------------------------------------------------------------------
// GB_select_iso: assign the iso value of C for GB_*selector
//------------------------------------------------------------------------------

static inline void GB_select_iso
(
    GB_void *Cx,                    // output iso value (same type as A)
    const GB_Opcode opcode,         // selector opcode
    const GB_void *athunk,          // thunk scalar, of size asize
    const GB_void *Ax,              // Ax [0] scalar, of size asize
    const size_t asize
)
{
    if (opcode == GB_VALUEEQ_idxunop_code)
    { 
        // all entries in C are equal to thunk
        memcpy (Cx, athunk, asize) ;
    }
    else
    { 
        // A and C are both iso
        memcpy (Cx, Ax, asize) ;
    }
}

//------------------------------------------------------------------------------
// compiler diagnostics
//------------------------------------------------------------------------------

// Some parameters are unused for some uses of the FactoryKernels/GB_sel_*
// functions
#include "GB_unused.h"

#endif

